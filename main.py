# In[] # import libraries 
from utils_process import peaks_on_flatten, choose_chunk_peak, calculate_50hz_fourier_coefficient, process_measurement, create_global_features
import time 
import pickle
import pyarrow 
import pyarrow.parquet as pq
import pandas as pd
from train import whole_process_training_single_iter, whole_process_training, whole_Network_training
from utils_model import display_metrics

import random
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt 


# In[] # settings 
preprocessed = True
recalculate_peaks = False 
input_dir = 'processed_input/'
signal_len = 800000
# window_sizes = [2000, 4000, 5000, 8000] 
window_size = 4000
nchunks = int(signal_len / window_size)

# In[] metadata
meta_df = pd.read_csv('metadata_train.csv')
signal_ids = meta_df['signal_id'].values


# =============================================================================
# ############ STEP 1 Preprocessing & Feature Extraction ######################
# =============================================================================
if not preprocessed:
    signal_df = pq.read_pandas(data_dir + 'train.parquet').to_pandas()
    _, all_flat_signals, all_points = peaks_on_flatten(signal_df, signal_ids)

    ##### STEP 1A. Denoise & Extract Waveforms #####
    # construct waveforms with given window size 
    waveforms = choose_chunk_peak(all_flat_signals, all_points, window_size=window_size)
    print(waveforms.shape)
    pickle.dump(waveforms, open(input_dir + 'all_chunk_waves_{}chunks.dat'.format(nchunks), 'wb'))

    ##### STEP 1B. Extract Global Features #####
    if recalculate_peaks:
        signal_fft = calculate_50hz_fourier_coefficient(signal_df.values)
        signal_peaks = process_measurement(signal_df, meta_df, signal_fft)
        signal_peaks.to_pickle(input_dir + 'signal_peaks.pkl')
        del signal_fft
        gc.collect()
    else:
        signal_peaks = pd.read_pickle(input_dir + 'signal_peaks.pkl')
    signal_peaks = pd.merge(signal_peaks, meta_df[['signal_id', 'id_measurement', 'target']], on='signal_id', how='left')

    ### load KMeans results
    # x, x_A, x_B, x_C = pickle.load(open('waves_list.dat','rb'))
    # kmeans = KMeans(n_clusters=15, random_state=9, init='k-means++').fit(x)
    # kmeans_A = KMeans(n_clusters=6, random_state=9, init='k-means++').fit(x_A)
    # kmeans_B = KMeans(n_clusters=6, random_state=9, init='k-means++').fit(x_B)
    # kmeans_C = KMeans(n_clusters=6, random_state=9, init='k-means++').fit(x_C)
    kmeans = pickle.load(open(input_dir + 'kmeans.dat', 'rb'))
    kmeans_A = pickle.load(open(input_dir + 'kmeans_A.dat', 'rb'))
    kmeans_B = pickle.load(open(input_dir + 'kmeans_B.dat', 'rb'))
    kmeans_C = pickle.load(open(input_dir + 'kmeans_C.dat', 'rb'))

    X_global = create_global_features(meta_df, signal_peaks, kmeans, kmeans_A, kmeans_B, kmeans_C)
    X_global.to_csv(input_dir + 'global_features.csv')

else:
    X_global = pd.read_csv(input_dir + 'global_features.csv')
    waveforms = pickle.load(open(input_dir + 'all_chunk_waves_{}chunks.dat'.format(nchunks), 'rb'))

X_global.set_index('id_measurement', inplace=True)


# =============================================================================
# ###################### STEP 2 Model Training ################################
# =============================================================================

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# In[] parameters
loss_name = 'weighted_bce'
output_folder = 'results_{}chunks_{}'.format(nchunks, loss_name)
# local_features = True 
load_local_features = True 
NN_level = 'signal'
NN_model = 'LSTM'
Dense_layers = 2
NN_pretrained = True 
layer_idx = 5 
NN_batch_size = 512 
classifier = 'XGboost'
classifier_level = 'measurement'
num_folds = 5 
num_iterations = 25 
feature_set = 'global'
kfold_random_state = 123948
pretrained = True 
predict = True 
weights_dict = None 
monitor = 'val_loss'
dropout = 0.4 
regularizer = 'l2'
from_logits = True
n_epochs = 100


########### LMFE ########### 
_, best_proba_LMFE, metrics_LMFE, test_pred_LMFE = whole_process_training(meta_df, waveforms, X_global,
    local_features=True, NN_level=NN_level, NN_model=NN_model,
    Dense_layers=Dense_layers, NN_pretrained=NN_pretrained, layer_idx=layer_idx, NN_batch_size=NN_batch_size, 
    output_folder=output_folder, classifier=classifier, classifier_level=classifier_level, num_folds=num_folds,
    num_iterations=num_iterations, feature_set=feature_set, kfold_random_state=kfold_random_state, 
    load_local_features=load_local_features, pretrained=pretrained, predict=predict, early_stopping_rounds=100, 
    verbose_eval=0, weights_dict=weights_dict, monitor=monitor, dropout=dropout, regularizer=regularizer, 
    loss_name=loss_name, from_logits=from_logits, n_epochs=n_epochs)
test_pred_LMFE.to_csv(output_folder + '/test_pred_LMFE.csv')
metrics_LMFE = display_metrics(test_pred_LMFE)

########### Only global-scale features ########### 
###### all global-scale features
# _, best_proba_global, metrics_global, test_pred_global = whole_process_training(meta_df, waveforms, X_global,
#     local_features=False, NN_level=NN_level, NN_model=NN_model, 
#     Dense_layers=Dense_layers, NN_pretrained=NN_pretrained, layer_idx=layer_idx, NN_batch_size=NN_batch_size, 
#     output_folder=output_folder, classifier='LightGBM', classifier_level=classifier_level, num_folds=num_folds,
#     num_iterations=num_iterations, feature_set='global', kfold_random_state=kfold_random_state, 
#     load_local_features=load_local_features, pretrained=pretrained, predict=predict, early_stopping_rounds=100, 
#     verbose_eval=0, weights_dict=weights_dict, monitor=monitor, dropout=dropout, regularizer=regularizer, 
#     loss_name=loss_name, from_logits=from_logits, n_epochs=n_epochs)
# test_pred_global.to_csv(output_folder + '/test_pred_global.csv')
test_pred_global = pd.read_csv(output_folder + '/test_pred_global.csv')
metrics_global = display_metrics(test_pred_global)

###### only phase-level features
# _, best_proba_phase, metrics_phase, test_pred_phase = whole_process_training(meta_df, waveforms, X_global,
#     local_features=False, NN_level=NN_level, NN_model=NN_model, 
#     Dense_layers=Dense_layers, NN_pretrained=NN_pretrained, layer_idx=layer_idx, NN_batch_size=NN_batch_size, 
#     output_folder=output_folder, classifier='LightGBM', classifier_level=classifier_level, num_folds=num_folds,
#     num_iterations=num_iterations, feature_set='phase_level', kfold_random_state=kfold_random_state, 
#     load_local_features=load_local_features, pretrained=pretrained, predict=predict, early_stopping_rounds=100, 
#     verbose_eval=0, weights_dict=weights_dict, monitor=monitor, dropout=dropout, regularizer=regularizer, 
#     loss_name=loss_name, from_logits=from_logits, n_epochs=n_epochs)
# test_pred_phase.to_csv(output_folder + '/test_pred_phase.csv')
test_pred_phase = pd.read_csv(output_folder + '/test_pred_phase.csv')
metrics_phase = display_metrics(test_pred_phase)

###### only measurement-level features
# _, best_proba_measure, metrics_measure, test_pred_measure = whole_process_training(meta_df, waveforms, X_global,
#     local_features=False, NN_level=NN_level, NN_model=NN_model, 
#     Dense_layers=Dense_layers, NN_pretrained=NN_pretrained, layer_idx=layer_idx, NN_batch_size=NN_batch_size, 
#     output_folder=output_folder, classifier='LightGBM', classifier_level=classifier_level, num_folds=num_folds,
#     num_iterations=num_iterations, feature_set='measurement_level', kfold_random_state=kfold_random_state, 
#     load_local_features=load_local_features, pretrained=pretrained, predict=predict, early_stopping_rounds=100, 
#     verbose_eval=0, weights_dict=weights_dict, monitor=monitor, dropout=dropout, regularizer=regularizer, 
#     loss_name=loss_name, from_logits=from_logits, n_epochs=n_epochs)
# test_pred_measure.to_csv(output_folder + '/test_pred_measure.csv')
test_pred_measure = pd.read_csv(output_folder + '/test_pred_measure.csv')
metrics_measure = display_metrics(test_pred_measure)

########### Only local-scale features ########### 
#### Prediction on test set
# _, best_proba_RNN, metrics_RNN, test_pred_RNN, _ = whole_Network_training(meta_df, waveforms,
#     NN_level=NN_level, NN_model=NN_model, Dense_layers=Dense_layers, NN_pretrained=NN_pretrained, 
#     layer_idx=layer_idx, NN_batch_size=NN_batch_size, indice_level=classifier_level,
#     output_folder=output_folder, kfold_random_state=kfold_random_state, num_folds=num_folds,
#     num_iterations=num_iterations, predict=predict, monitor=monitor, dropout=dropout, regularizer=regularizer,
#     from_logits=from_logits, loss_name=loss_name, extract_attention_weights=False)
# test_pred_RNN.to_csv(output_folder + '/test_pred_RNN.csv')
# load test set prediction
test_pred_RNN = pd.read_csv(output_folder + '/test_pred_RNN.csv')
metrics_RNN = display_metrics(test_pred_RNN)

# In[] display the performance
all_metrics = np.array([metrics_LMFE, metrics_global, metrics_phase, metrics_measure, metrics_RNN])
df_res = pd.DataFrame(data=all_metrics, index=['LMFE', 'Global-scale', 'Phase', 'Measurement', 'RNN'], 
    columns=['MCC', 'Precision', 'Recall', 'F1 Score', 'AUC'])
print(df_res)


# =============================================================================
# ###################### STEP 3 Model Evaluation ##############################
# =============================================================================
load_attention_weights = True

# In[] extract attention weights
if not load_attention_weights:
    _, _, _, _, attention_weights = whole_Network_training(meta_df, waveforms,
            NN_level=NN_level, NN_model=NN_model, Dense_layers=Dense_layers, NN_pretrained=NN_pretrained, 
            layer_idx=layer_idx, NN_batch_size=NN_batch_size, indice_level=classifier_level,
            output_folder=output_folder, kfold_random_state=kfold_random_state, num_folds=num_folds,
            num_iterations=num_iterations, predict=False, monitor=monitor, dropout=dropout, regularizer=regularizer,
            kernel_size=kernel_size, from_logits=from_logits, loss_name=loss_name, extract_attention_weights=True)

    pickle.dump(attention_weights.squeeze(), open(output_folder + '/attention_weights_{}_{}Dense_layers_{}_level.dat'.format(NN_model, 
        Dense_layers, NN_level), 'wb'))
else:
    attention_weights = pickle.load(open(output_folder + '/attention_weights_{}_{}Dense_layers_{}_level.dat'.format(NN_model, 
        Dense_layers, NN_level), 'rb'))

### Obtain representative waveforms with largest weights for each signal
max_weights_loc = np.argmax(attention_weights.squeeze(), 1) # length N
weighted_waves = np.zeros((waveforms.shape[0], waveforms.shape[2]))
for i in range(len(max_weights_loc)):
    weighted_waves[i, :] = waveforms[i, max_weights_loc[i], :]
normalized_weighted_waves = weighted_waves * np.sign(weighted_waves[:, 15]).reshape(-1,1) / np.max(abs(weighted_waves), axis=1).reshape(-1,1)

### Clustering for all faulty signals
y_data = meta_df['target'].values
normalized_faulty_waves = normalized_weighted_waves[y_data==1, :] # (525,30)
NA = np.isnan(normalized_faulty_waves)[:, 0]
new_faulty = normalized_faulty_waves[~NA, :]

kmeans_faulty = KMeans(n_clusters=6, random_state=9, init='k-means++').fit(new_faulty)
### visualization
y_faulty = kmeans_faulty.predict(new_faulty)
centroid_faulty = kmeans_faulty.cluster_centers_
colors = ['r','y','g','b','m','c']
fig, ax = plt.subplots(3,2,figsize=(16,8))
for i in range(6):
    xx = new_faulty[y_faulty == i, ]
    ax[i//2,i%2].plot(xx[:100,].T, color=colors[i], linestyle=':',linewidth=0.2)
    ax[i//2,i%2].plot(centroid_faulty[i,], color='k')
    ax[i//2,i%2].set_ylim(-1,1)
    ax[2,i%2].set_xlabel('Time Step')
    ax[i//2, 0].set_ylabel('Normalized \nAmplitude')
plt.show()


