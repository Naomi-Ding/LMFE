# In[] Import all the libraries 
import tensorflow as tf 
from tensorflow.keras.layers import Input, LSTM, Bidirectional, GRU, Dense, Layer, Dropout, Conv1D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import activations, regularizers, initializers, constraints, optimizers 
from tensorflow.keras import backend as K
from tensorflow.keras import Model 
from minimal_rnn_tf import MinimalRNN 
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score

import numpy as np
import pandas as pd
import pickle

# =============================================================================
# ######################## Utils for RNN model  ###############################
# =============================================================================

# Matthews correlation coefficient calculation used inside Keras model
def matthews_correlation(y_true, y_pred):
  """
  Calculate Matthews Correlation Coefficient.

  References
  ----------
  .. [1] https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
  .. [2] https://www.kaggle.com/tarunpaparaju/vsb-competition-attention-bilstm-with-features/notebook?scriptVersionId=10690570
  """
  y_pred_pos = K.round(K.clip(y_pred, 0, 1))
  y_pred_neg = 1 - y_pred_pos

  y_pos = K.round(K.clip(y_true, 0, 1))
  y_neg = 1 - y_pos

  tp = K.sum(y_pos * y_pred_pos)
  tn = K.sum(y_neg * y_pred_neg)

  fp = K.sum(y_neg * y_pred_pos)
  fn = K.sum(y_pos * y_pred_neg)

  numerator = (tp * tn - fp * fn)
  denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

  return numerator / (denominator + K.epsilon())

# In[] Attention Layer & RNN architecture
class Attention(Layer):
  # https://keras.io/layers/writing-your-own-keras-layers/
  # class Attention(Layer):
  #   """
  #   Performs basic attention layer operation.

  #   References
  #   ----------
  #   .. [1] https://arxiv.org/pdf/1512.08756.pdf
  #   .. [2] https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
  #   .. [3] https://www.kaggle.com/tarunpaparaju/vsb-competition-attention-bilstm-with-features/notebook?scriptVersionId=10690570
  #   """
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
    # https://keras.io/regularizers/
    # Define weight and bias regularizer    
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
    # https://keras.io/constraints/
    # Define weight and bias constraints
    # Contraints => to keep check on the weight and bias values    
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
    
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
      """
      Build the Attention Layer.
      """
      assert len(input_shape) == 3
    
      # add_weight() comes from keras.layers.add_weight()
      self.W = self.add_weight(shape=(input_shape[-1],), initializer=self.init, 
                               name="{}_W".format(self.name), 
                               regularizer=self.W_regularizer, 
                               constraint=self.W_constraint)    
      self.features_dim = input_shape[-1]
    
      if self.bias:
        self.b = self.add_weight(shape=(input_shape[1],),
                                 initializer='zero', 
                                 name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer,
                                 constraint=self.b_constraint)
      else:
        self.b = None
      self.built = True
    
    def compute_mask(self, input, input_mask=None):
      """
      Do not pass the mask to the next layer.
      """
      return None
    
    def call(self, x, mask=None):
      """
      Performs attention mechanism.
      """
      features_dim = self.features_dim
      step_dim = self.step_dim
      
      # https://keras.io/backend/#reshape
      # K.reshape(x, shape)
      # x -> tensor or variable to be reshaped
      # shape -> target shape
      # K.reshape(x, (-1,cols)) -> will reshape the variable x according to the given columns, no. of rows needed is adjusted accordingly
      # Get the dot product of (x,self.W) and reshape the dot product of (x,self.W) to have step_dim no. of columns and rows are adjusted accordingly
      eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
    
      if self.bias:
        eij += self.b
    
      eij = K.tanh(eij)
    
      # https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
      a = K.exp(eij)
    
      if mask is not None:
        # typecast mask to a 32-bit float value
        a *= K.cast(mask, K.floatx())
    
      # Perform softmax operation
      a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
    
      a = K.expand_dims(a)
      weighted_input = x * a
      return K.sum(weighted_input, axis=1), a
    
    def compute_output_shape(self, input_shape):
      """
      Compute the shape of the output.
      """
      return input_shape[0], self.features_dim


def weighted_binary_cross_entropy(weights: dict, from_logits: bool = True):
    assert 0 in weights
    assert 1 in weights

    def weighted_cross_entropy_fn(y_true, y_pred):
        tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
        tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
        ce = K.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
        loss = K.mean(tf.multiply(ce, tf.cast(weights_v, dtype=ce.dtype)))
        return loss

    return weighted_cross_entropy_fn

def model_lstm(input_shape, Dense_layers=2, dropout=0.2, regularizer='l2', loss_name='bce', weights={0:0.1,1:0.9}, from_logits=True):
  """
  Builds the Neural Network Architecture.

  Following is the architecture that is built:
  * Layer 1
      * LSTM
        * Bidirectional LSTM - 128 neurons
        * Bidirectional LSTM - 64 neurons
      * Attention layer
  * Layer 2
      * Dense - 64, activation: relu
  * Layer 3
    * Output: Dense - 1, activation: sigmoid
  * Loss - binary cross-entropy
  * Optimizer - adam
  * Metric - matthews correlation coefficient
  """
  inp = Input(shape=(input_shape[1], input_shape[2],))
  bi_lstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=dropout), merge_mode='concat')(inp)
  bi_lstm_2 = Bidirectional(GRU(64, return_sequences=True, dropout=dropout), merge_mode='concat')(bi_lstm_1)
  attention = Attention(input_shape[1])(bi_lstm_2)[0]
  x = Dropout(dropout)(attention)
  # x = concatenate([attention, feat], axis=1)
  if Dense_layers > 1:
      x = Dense(64, activation='relu', kernel_regularizer=regularizer)(x)
      x = Dropout(dropout)(x)
      if Dense_layers == 3:
          x = Dense(32, activation='relu', kernel_regularizer=regularizer)(x)
          x = Dropout(dropout)(x)
      x = Dense(1, activation='linear')(x)
  elif Dense_layers == 1:
      x = Dense(1, activation='linear')(x)

  model = Model(inputs=inp, outputs=x)
  # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
  if loss_name == 'bce':
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
  elif loss_name == 'weighted_bce':
    loss = weighted_binary_cross_entropy(weights, from_logits)
  elif loss_name == 'focal':
    loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=from_logits)
  model.compile(loss=loss, optimizer='adam', metrics=[matthews_correlation])

  return model


def model_minimalRNN(input_shape, Dense_layers=2, dropout=0.2, regularizer='l2', loss_name='bce', weights={0:0.1,1:0.9}, from_logits=True):
  """
  Builds the Neural Network Architecture based on minimalRNN.

  Following is the architecture that is built:
  * Layer 1
      * MinimalRNN - 128 neurons
      * Attention layer
  * Layer 2
      * Dense - 64, activation: relu
  * Layer 3
    * Output: Dense - 1, activation: sigmoid
  * Loss - binary cross-entropy
  * Optimizer - adam
  * Metric - matthews correlation coefficient
  """

  inp = Input(shape=(input_shape[1], input_shape[2],))
  mini_rnn = MinimalRNN(units=128, use_bias=False, return_sequences=True)(inp)
  attention = Attention(input_shape[1])(mini_rnn)[0]
  x = Dropout(dropout)(attention)
  # x = concatenate([attention, feat], axis=1)
  if Dense_layers > 1:
      x = Dense(64, activation='relu', kernel_regularizer=regularizer)(x)
      x = Dropout(dropout)(x)
      if Dense_layers == 3:
          x = Dense(32, activation='relu', kernel_regularizer=regularizer)(x)
          x = Dropout(dropout)(x)
      x = Dense(1, activation='linear')(x)
  elif Dense_layers == 1:
      x = Dense(1, activation='linear')(x)

  model = Model(inputs=inp, outputs=x)
  # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
  if loss_name == 'bce':
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
  elif loss_name == 'weighted_bce':
    loss = weighted_binary_cross_entropy(weights, from_logits)
  elif loss_name == 'focal':
    loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=from_logits)
  model.compile(loss=loss, optimizer='adam', metrics=[matthews_correlation])

  return model


def model_temporalCNN(input_shape, kernel_size=[12,7], loss_name='bce', weights={0:0.1,1:0.9}, from_logits=True):
  # input_shape = (100, 200, 40, 1)
  inp = Input(shape=(input_shape[1], input_shape[2], input_shape[3],))
  cnn1 = Conv1D(16, kernel_size[0], padding='same', activation='relu')(inp)
  cnn2 = Conv1D(16, kernel_size[0], padding='same', activation='relu')(cnn1)
  map1 = MaxPooling2D((1,2))(cnn2)

  cnn3 = Conv1D(8, kernel_size[1], padding='same', activation='relu')(map1)
  cnn4 = Conv1D(8, kernel_size[1], padding='same', activation='relu')(cnn3)
  map2 = MaxPooling2D((1,2))(cnn4)

  gap = GlobalAveragePooling2D()(map2)
  x = Dense(32, activation='relu')(gap)
  x = Dense(1, activation='linear')(x)

  model = Model(inputs=inp, outputs=x)
  if loss_name == 'bce':
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
  elif loss_name == 'weighted_bce':
    loss = weighted_binary_cross_entropy(weights, from_logits)
  elif loss_name == 'focal':
    loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=from_logits)
  model.compile(loss=loss, optimizer='adam', metrics=[matthews_correlation])

  return model

  
# In[] combine signal-level waveforms for each measurement 
def combine_measure_waves(signal_waves):

    assert signal_waves.shape[0] % 3 == 0
    N = signal_waves.shape[0]//3
    L = signal_waves.shape[2]
    measure_waves = np.zeros([N, signal_waves.shape[1], 3*L])
    
    for i in range(N):
        sigids = [i*3, i*3+1, i*3+2]
        measure_wave = signal_waves[sigids, :,:] # 3x160x30
        for j in range(3):
            measure_waves[i, :, j*L:(j+1)*L] = measure_wave[j, :,:]
            
    return measure_waves 


# In[] combine signal-level intermediate features for each measurement 
def combine_measure_inter_feature(signal_inter_features):
    assert signal_inter_features.shape[0] % 3 == 0
    N = signal_inter_features.shape[0] // 3
    L = signal_inter_features.shape[1]
    measure_inter_features = np.zeros([N, 3*L])
    for i in range(N):
        sigids = [i*3, i*3+1, i*3+2]
        measure_inter_feature = signal_inter_features[sigids, :] # 3xL
        for j in range(3):
            measure_inter_features[i, j*L:(j+1)*L] = measure_inter_feature[j, :]
            
    return measure_inter_features


# In[] combine all the local-scale features & global features
def combine_global_features(meta_train_df, global_features, classifier_level='measurement', feature_set='global'):

    # global_features = pd.read_csv(global_data_dir + 'X_train.csv').set_index('id_measurement')
    # num_feature = inter_feature_train.shape[1]

    feature_names_measurement_level = [
    'peak_count_Q13',
    'height_mean_Q02', 
    'height_std_Q02',
    'sawtooth_rmse_mean_Q02',
    'sawtooth_rmse1_mean_Q02',
    'sawtooth_rmse2_mean_Q02',
    'sawtooth_rmse3_mean_Q02',
    'sawtooth_rmse4_mean_Q02',
    'sawtooth_rmse5_mean_Q02',       
    'sawtooth_rmse6_mean_Q02', 
    'sawtooth_rmse7_mean_Q02', 
    'peak_count_Q02',
    'peak_count_total',
    'number_peaks_0', 'number_peaks_1', 'number_peaks_2', 'number_peaks_3', 'number_peaks_4', 'number_peaks_5', 'number_peaks_6', 'number_peaks_7', 'number_peaks_8', 'number_peaks_9', 'number_peaks_10', 'number_peaks_11', 'number_peaks_12', 'number_peaks_13', 'number_peaks_14',
    'mean_height_peaks_0', 'mean_height_peaks_1', 'mean_height_peaks_2', 'mean_height_peaks_3', 'mean_height_peaks_4', 'mean_height_peaks_5', 'mean_height_peaks_6', 'mean_height_peaks_7', 'mean_height_peaks_8', 'mean_height_peaks_9', 'mean_height_peaks_10', 'mean_height_peaks_11', 'mean_height_peaks_12', 'mean_height_peaks_13', 'mean_height_peaks_14',
    'std_height_peaks_0', 'std_height_peaks_1', 'std_height_peaks_2', 'std_height_peaks_3', 'std_height_peaks_4', 'std_height_peaks_5', 'std_height_peaks_6', 'std_height_peaks_7', 'std_height_peaks_8', 'std_height_peaks_9', 'std_height_peaks_10', 'std_height_peaks_11', 'std_height_peaks_12', 'std_height_peaks_13', 'std_height_peaks_14', 
    'mean_RMSE_peaks_0', 'mean_RMSE_peaks_1', 'mean_RMSE_peaks_2', 'mean_RMSE_peaks_3', 'mean_RMSE_peaks_4', 'mean_RMSE_peaks_5', 'mean_RMSE_peaks_6', 'mean_RMSE_peaks_7', 'mean_RMSE_peaks_8', 'mean_RMSE_peaks_9', 'mean_RMSE_peaks_10', 'mean_RMSE_peaks_11', 'mean_RMSE_peaks_12', 'mean_RMSE_peaks_13', 'mean_RMSE_peaks_14', 
    ]

    feature_names_phase_level = [
    'number_peaks_A0', 'number_peaks_B0', 'number_peaks_C0',
    'number_peaks_A1', 'number_peaks_B1', 'number_peaks_C1',
    'number_peaks_A2', 'number_peaks_B2', 'number_peaks_C2',
    'number_peaks_A3', 'number_peaks_B3', 'number_peaks_C3',
    'number_peaks_A4', 'number_peaks_B4', 'number_peaks_C4',
    'number_peaks_A5', 'number_peaks_B5', 'number_peaks_C5',
    'mean_height_peaks_A0', 'mean_height_peaks_B0', 'mean_height_peaks_C0',
    'mean_height_peaks_A1', 'mean_height_peaks_B1', 'mean_height_peaks_C1',
    'mean_height_peaks_A2', 'mean_height_peaks_B2', 'mean_height_peaks_C2', 
    'mean_height_peaks_A3', 'mean_height_peaks_B3', 'mean_height_peaks_C3',
    'mean_height_peaks_A4', 'mean_height_peaks_B4', 'mean_height_peaks_C4',
    'mean_height_peaks_A5', 'mean_height_peaks_B5', 'mean_height_peaks_C5',       
    'std_height_peaks_A0', 'std_height_peaks_B0', 'std_height_peaks_C0',
    'std_height_peaks_A1', 'std_height_peaks_B1', 'std_height_peaks_C1',
    'std_height_peaks_A2', 'std_height_peaks_B2', 'std_height_peaks_C2',
    'std_height_peaks_A3', 'std_height_peaks_B3', 'std_height_peaks_C3',
    'std_height_peaks_A4', 'std_height_peaks_B4', 'std_height_peaks_C4',
    'std_height_peaks_A5', 'std_height_peaks_B5', 'std_height_peaks_C5',
    'mean_RMSE_peaks_A0', 'mean_RMSE_peaks_B0', 'mean_RMSE_peaks_C0',
    'mean_RMSE_peaks_A1', 'mean_RMSE_peaks_B1', 'mean_RMSE_peaks_C1',
    'mean_RMSE_peaks_A2', 'mean_RMSE_peaks_B2', 'mean_RMSE_peaks_C2',
    'mean_RMSE_peaks_A3', 'mean_RMSE_peaks_B3', 'mean_RMSE_peaks_C3',
    'mean_RMSE_peaks_A4', 'mean_RMSE_peaks_B4', 'mean_RMSE_peaks_C4',
    'mean_RMSE_peaks_A5', 'mean_RMSE_peaks_B5', 'mean_RMSE_peaks_C5',
    ]

    if classifier_level == 'measurement':
        X_train = global_features.copy()
        if feature_set == 'measurement_level':
            X_train = X_train[feature_names_measurement_level]
        elif feature_set == 'phase_level':
            X_train = X_train[feature_names_phase_level]
        elif feature_set == 'global':
            feature_names = [c for c in global_features.columns]   

        y_train = (meta_train_df.groupby('id_measurement')['target'].sum().round(0).astype(np.int)!= 0).astype(np.float)
        
    elif classifier_level == 'signal':
        if feature_set == 'measurement_level':
            feature_names = feature_names_measurement_level
        elif feature_set == 'phase_level':
            feature_names = feature_names_phase_level
        elif feature_set == 'global':
            feature_names = [c for c in global_features.columns]   

        feature_A = [s for s in feature_names if 'A' in s]
        feature_B = [s for s in feature_names if 'B' in s]
        feature_C = [s for s in feature_names if 'C' in s]
        common_feature = [c for c in feature_names if c not in feature_A + feature_B + feature_C]
        feature_names = [c.replace('A','') for c in feature_A] + ['measurement_'+c for c in common_feature]

        feature_train_A = global_features.loc[:, feature_A + common_feature]
        feature_train_B = global_features.loc[:, feature_B + common_feature]
        feature_train_C = global_features.loc[:, feature_C + common_feature]

        feature_train_A['signal_id'] = np.arange(0, meta_train_df.shape[0], 3)
        feature_train_B['signal_id'] = np.arange(1, meta_train_df.shape[0], 3)
        feature_train_C['signal_id'] = np.arange(2, meta_train_df.shape[0], 3)
        feature_train_A.columns= list(feature_names + ['signal_id'])
        feature_train_B.columns= list(feature_names + ['signal_id'])
        feature_train_C.columns= list(feature_names + ['signal_id'])

        X_train = pd.concat([feature_train_A, feature_train_B, feature_train_C])
        signal_ids = meta_train_df['signal_id']
        X_train = X_train.reset_index(drop=True).set_index('signal_id').reindex(signal_ids)

        y_train = meta_train_df['target'].astype(np.float)


    assert np.all(y_train.index.values == X_train.index.values)
    feature_names = [c for c in X_train.columns]
    # print('Shape of X_train: {}'.format(X_train.shape))

    return X_train, y_train, feature_names


# In[] combine the global features & intermediate feautres 
def combine_intermediate_features(global_features, inter_features,
    NN_level='signal', classifier_level='measurement'):
    
    X_data = global_features.copy()

    if classifier_level == 'measurement':
        if NN_level == 'signal':
            measure_inter_features = combine_measure_inter_feature(inter_features)
        elif NN_level == 'measurement':
            measure_inter_features = inter_features

        for i in range(measure_inter_features.shape[1]):
            feature_name = 'RNN_feature_' + str(i)
            X_data[feature_name] = measure_inter_features[:, i]   

        # y_data = (meta_train_df.groupby('id_measurement')['target'].sum().round(0).astype(np.int)!= 0).astype(np.float)
    
    elif classifier_level == 'signal':
        assert NN_level == 'signal' # RNN must be trained on signal level
        for i in range(inter_features.shape[1]):
            feature_name = 'RNN_feature_' + str(i)
            X_data[feature_name] = inter_features[:, i]

        # y_data = meta_train_df['target'].astype(np.float)

    feature_names = [c for c in X_data.columns]       
    # assert np.all(y_data.index.values == X_data.index.values)
    # print('Shape of X_data: {}'.format(X_data.shape))

    return X_data, feature_names


# =============================================================================
# ###################### Utils for Results Analysis  ##########################
# =============================================================================
# In[] feature importance analysis
def feature_importance(meta_train_df, global_features, classifier='XGboost', feature_set='global',
  output_folder='results_v3/nchunks_160', loss_name='weighted_bce', NN_model='LSTM', Dense_layers=2, NN_level='signal', 
  monitor='val_loss', num_feature=64, classifier_level='measurement', importance_type='total_gain', num_iterations=25,
  num_folds=5):

  _, _, feature_names = combine_global_features(meta_train_df, global_features, classifier_level=classifier_level, feature_set=feature_set)
  if classifier_level == 'measurement':
    rnn_features = num_feature * 3
  elif classifier_level == 'signal':
    rnn_features = num_feature
  for i in range(rnn_features):
    rnn_name = 'RNN_feature_' + str(i)
    feature_names.append(rnn_name)

  importances = pd.DataFrame()
  models = []
  # for iter in tqdm(range(num_iterations)):
  for iter in range(num_iterations):
    for fold in range(num_folds):
      model_file_name = '{}/classifier/{}/{}_{}Dense_layers_{}_level_monitor_{}_{}interfeatures_{}_{}_level_iter{}_fold{}.dat'.format(output_folder, \
                  loss_name, NN_model, Dense_layers, NN_level, monitor, num_feature, classifier, classifier_level, iter, fold)
      model = pickle.load(open(model_file_name, 'rb'))
      model.feature_names = feature_names
      models.append(model)

      imp_df = pd.DataFrame()
      if classifier == 'XGboost':
        imp = model.get_score(importance_type=importance_type)
        imp_df['feature'] = imp.keys()
        imp_df['gain'] = imp.values()
      elif classifier == 'LightGBM':
        imp_df['feature'] = feature_names        
        imp_df['gain'] = model.feature_importance('gain')
      imp_df['fold'] = num_folds * iter + fold + 1 
      importances = pd.concat([importances, imp_df], axis=0, sort=False)

  important_features = importances[['gain', 'feature']].groupby('feature').mean().sort_values('gain', ascending=False)
  imp_file_name = '{}/feature_importance_{}_{}_{}Dense_layers_{}_level_{}interfeatures_{}_{}_level'.format(output_folder, \
              importance_type, NN_model, Dense_layers, NN_level, num_feature, classifier, classifier_level)
  important_features.to_csv('{}.csv'.format(imp_file_name))
  models_name = '{}/all_models_{}_{}Dense_layers_{}_level_{}interfeatures_{}_{}_level.dat'.format(output_folder, \
              NN_model, Dense_layers, NN_level, num_feature, classifier, classifier_level)
  pickle.dump(models, open(models_name, 'wb'))
  return important_features


# In[] Analysis of Classification Performance
# Find optimal threshold 
def optimal_threshold(y_true, y_prob):
    thresholds = np.linspace(0,1,100)[1:]
    scores = [] 

    for t in thresholds:
        s_val = matthews_corrcoef(y_true, y_prob > t)
        scores.append(s_val)
    
    best_mcc = np.max(scores)
    best_proba = thresholds[np.argmax(scores)]
    return best_proba, best_mcc

def performance_analysis(meta_train_df, yp_train, yp_val, yp_test, predict_level='measurement'):
    if predict_level == 'measurement':
        train_pred = meta_train_df[['id_measurement', 'signal_id', 'target']].copy()
        yp_train_df = pd.DataFrame(yp_train,index=meta_train_df['id_measurement'].unique())
        yp_train_df.index.rename('id_measurement', inplace=True)
        train_pred = pd.merge(train_pred, yp_train_df, on='id_measurement')
        train_pred.rename({0:'prediction'}, axis=1, inplace=True)

        val_pred = meta_train_df[['id_measurement', 'signal_id', 'target']].copy()
        yp_val_df = pd.DataFrame(yp_val,index=meta_train_df['id_measurement'].unique())
        yp_val_df.index.rename('id_measurement', inplace=True)
        val_pred = pd.merge(val_pred, yp_val_df, on='id_measurement')
        val_pred.rename({0:'prediction'}, axis=1, inplace=True)

        test_pred = meta_train_df[['id_measurement', 'signal_id', 'target']].copy()
        yp_test_df = pd.DataFrame(yp_test,index=meta_train_df['id_measurement'].unique())
        yp_test_df.index.rename('id_measurement', inplace=True)
        test_pred = pd.merge(test_pred, yp_test_df, on='id_measurement')
        test_pred.rename({0:'prediction'}, axis=1, inplace=True)

        best_proba, best_mcc_val = optimal_threshold(val_pred['target'].values.astype(np.float), val_pred['prediction'].values.astype(np.float))
        best_mcc_test = matthews_corrcoef(test_pred['target'].values.astype(np.float), test_pred['prediction'].values.astype(np.float) > best_proba)
        test_pred['probability_thresholded'] = (test_pred['prediction'] > best_proba).astype(np.int)
        best_mcc_train = matthews_corrcoef(train_pred['target'].values.astype(np.float), train_pred['prediction'].values.astype(np.float) > best_proba)
 
    elif predict_level == 'signal':
        y_train = meta_train_df['target'].values
        best_proba, best_mcc_val = optimal_threshold(y_train.astype(np.float), yp_val.astype(np.float))
        best_mcc_test = matthews_corrcoef(y_train.astype(np.float), yp_test.astype(np.float) > best_proba)
        best_mcc_train = matthews_corrcoef(y_train.astype(np.float), yp_train.astype(np.float) > best_proba)

        test_pred = meta_train_df[['id_measurement', 'signal_id', 'target']].copy()
        test_pred['prediction'] = yp_test 
        test_pred['probability_thresholded'] = (yp_test > best_proba).astype(np.int)

    ##################################
    print("Best Probability Threshold based on validation set: {:.3f}".format(best_proba))
    print("MCC Training: {:.3f}".format(best_mcc_train))
    print("MCC Validation: {:.3f}".format(best_mcc_val))
    print("MCC Test: {:.3f}".format(best_mcc_test))

    y_true = test_pred['target']
    y_pred = test_pred['probability_thresholded']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)

    print('For best probability thresholded: {:.3f},\n \
        mcc:{}, precision:{}, recall:{}, f1:{}, acc:{}, roc_auc:{},average_precision:{}\n'.format(best_proba,\
            mcc, precision, recall, f1, acc, roc_auc, average_precision))

    return best_proba, [mcc, precision, recall, f1, acc, roc_auc, average_precision], test_pred


def display_metrics(test_pred):
  y_true = test_pred['target']
  y_pred = test_pred['probability_thresholded']
  mcc = matthews_corrcoef(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  roc_auc = roc_auc_score(y_true, y_pred)

  return [mcc, precision, recall, f1, roc_auc]