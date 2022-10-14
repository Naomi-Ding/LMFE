import pickle
import numpy as np
import pandas as pd
from utils_model import combine_measure_waves, combine_global_features, combine_intermediate_features, performance_analysis
from utils_model import Attention, model_lstm, model_minimalRNN, model_temporalCNN
from tensorflow.keras.callbacks import * 
from tensorflow.keras import Model 
from tensorflow.keras import backend as K 
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
import warnings 
from tqdm import tqdm

# =============================================================================
# ################# NN Training & Extract intermediate features ###############
# =============================================================================
def Network_Training_single_fold(meta_df, signal_waves, signal_y, train_indices, 
	val_indices, test_indices, indice_level = 'measurement',
	NN_level='signal', NN_model='LSTM', Dense_layers=2, NN_pretrained=False, 
	ckpt_name='results_200chunks_weighted_bce/_LSTM_2Dense_layers_signal_level_iter0_fold0.h5', 
	predict=True, intermediate_features=False, layer_idx=5, batch_size=512, monitor='val_loss',
	dropout=0.4, regularizer='l2', loss_name='weighted_bce', from_logits=True, weights_dict=None, 
	kernel_size=[12,7], n_epochs=100, extract_attention_weights=False):

	### train_indices, val_indices, test_indices are all measurement-level, so need to be adjusted first 
	if indice_level == 'measurement':
		if NN_level == 'signal':
			train_indices = np.where(meta_df['id_measurement'].isin(train_indices))[0]
			val_indices = np.where(meta_df['id_measurement'].isin(val_indices))[0]
			test_indices = np.where(meta_df['id_measurement'].isin(test_indices))[0]
		elif NN_level == 'measurement':
			signal_waves = combine_measure_waves(signal_waves)
			signal_y = (meta_df.groupby('id_measurement')['target'].sum().round(0).astype(np.int)!=0).astype(np.float)
	else:
		# if train_indices, val_indices, test_indices are all signal-level, 
		# NN_level must be signal, then no need to process indices and waveforms 
		assert NN_level == 'signal'

	train_X, train_y = signal_waves[train_indices], signal_y[train_indices]
	val_X, val_y = signal_waves[val_indices], signal_y[val_indices] 
	test_X, test_y = signal_waves[test_indices], signal_y[test_indices]

	if loss_name == 'weighted_bce' and weights_dict is None:
		weights = len(train_y) / (np.bincount(train_y) * 2)
		weights_dict = {0: weights[0], 1:weights[1]}

	if NN_model == 'LSTM':
	    model = model_lstm(signal_waves.shape, Dense_layers, dropout=dropout, regularizer=regularizer, loss_name=loss_name, weights=weights_dict, from_logits=from_logits)
	elif NN_model == 'minimal_rnn':
	    model = model_minimalRNN(signal_waves.shape, Dense_layers, dropout=dropout, regularizer=regularizer, loss_name=loss_name, weights=weights_dict, from_logits=from_logits)
	elif NN_model == 'TCN':
		signal_waves = signal_waves[..., np.newaxis]
		model = model_temporalCNN(signal_waves.shape, kernel_size=kernel_size, loss_name=loss_name, weights=weights_dict)

	if not NN_pretrained:
		ckpt = ModelCheckpoint(ckpt_name, save_best_only=True, save_weights_only=True,
	   		verbose=2, monitor=monitor, mode='min')
		model.fit(train_X, train_y, batch_size=batch_size, epochs=n_epochs, validation_data=(val_X, val_y), \
	    	callbacks=[ckpt], verbose=2) 

	model.load_weights(ckpt_name)

	if predict:
		yp = model.predict(train_X, batch_size=512)
		yp_val_fold = model.predict(val_X, batch_size=512)
		yp_test_fold = model.predict(test_X, batch_size=512)
	else:
		yp, yp_val_fold, yp_test_fold = None, None, None

	if intermediate_features:
	    inter_model = Model(inputs = model.input, outputs = model.get_layer(index=layer_idx).output)
	    inter_features = inter_model.predict(signal_waves) 
	else:
		inter_features = None

	if extract_attention_weights:
		inter_model2 = Model(inputs = model.input, outputs = model.get_layer(index=2).output)
		inter_features_train = inter_model2.predict(train_X)
		weight = Attention(signal_waves.shape[1])(inter_features_train)[1] # (60%*N, nchunks, 1)
	else:
		weight = None

	return yp, yp_val_fold, yp_test_fold, inter_features, weight


def whole_Network_training(meta_df, signal_waves, NN_level='signal', NN_model='LSTM', nchunks=200,
	Dense_layers=2, NN_pretrained=False, layer_idx = 5, NN_batch_size = 512, indice_level='measurement',
	output_folder='results_200chunks_weighted_bce', kfold_random_state=123948, 
	num_folds=5, num_iterations=25, predict=True, monitor='val_loss', weights_dict=None,
    dropout=0.4, regularizer='l2', loss_name='bce', from_logits=True, kernel_size=[12,7], 
    n_epochs=100, extract_attention_weights=False):

	signal_y = meta_df['target'].values
	measure_y = (meta_df.groupby('id_measurement')['target'].sum().round(0).astype(np.int)!= 0).astype(np.float)

	if predict:
		if NN_level == 'signal':
		    yp_train = np.zeros(signal_y.shape[0])
		    yp_val = np.zeros(signal_y.shape[0])
		    yp_test = np.zeros(signal_y.shape[0])
		elif NN_level == 'measurement':
		    yp_train = np.zeros(measure_y.shape[0])
		    yp_val = np.zeros(measure_y.shape[0])
		    yp_test = np.zeros(measure_y.shape[0])
	else:
	    yp_train = None
	    yp_val = None
	    yp_test = None
	    best_proba, metrics, test_pred = None, None, None 

	if extract_attention_weights:
		attention_weights = np.zeros([signal_y.shape[0], nchunks, 1])
	else:
		attention_weights = None


	for iter in tqdm(range(num_iterations)):
		##### split the dataset 
		np.random.seed(kfold_random_state + iter)
		splits = np.zeros(measure_y.shape[0], dtype=np.int)
		m = measure_y == 1
		splits[m] = np.random.randint(0, 5, size=m.sum())
		m = measure_y == 0
		splits[m] = np.random.randint(0, 5, size=m.sum())

		# for fold in tqdm(range(num_folds)):
		for fold in range(num_folds):
			# print("Beginning iteration {}, fold {}".format(iter, fold))        	
			val_fold = fold
			test_fold = (fold + 1) % num_folds
			train_folds = [f for f in range(num_folds) if f not in [val_fold, test_fold]]

			train_indices = np.where(np.isin(splits, train_folds))[0]
			val_indices = np.where(splits == val_fold)[0]
			test_indices = np.where(splits == test_fold)[0]

			K.clear_session()
			# print("NN Training & Extracting intermediate features")
			ckpt_name = '{}/RNN_weights/{}_{}Dense_layers_{}_level_monitor_{}_iter{}_fold{}.h5'.format(output_folder, \
				NN_model, Dense_layers, NN_level, monitor, iter, fold)
			yp, yp_val_fold, yp_test_fold, _, weight = Network_Training_single_fold(meta_df, signal_waves, 
				signal_y, train_indices, val_indices, test_indices, indice_level=indice_level, 
				NN_level=NN_level, NN_model=NN_model, Dense_layers=Dense_layers, NN_pretrained=NN_pretrained,
				ckpt_name=ckpt_name, predict=predict, intermediate_features=False, layer_idx=layer_idx, 
				batch_size=NN_batch_size, monitor=monitor, dropout=dropout, regularizer=regularizer, loss_name=loss_name,
				kernel_size=kernel_size, n_epochs=n_epochs, weights_dict=weights_dict, from_logits=from_logits,
				extract_attention_weights=extract_attention_weights)

			if NN_level == 'signal':
				train_indices = np.where(meta_df['id_measurement'].isin(train_indices))[0]
				val_indices = np.where(meta_df['id_measurement'].isin(val_indices))[0]
				test_indices = np.where(meta_df['id_measurement'].isin(test_indices))[0]

			if predict:
				yp_train[train_indices] += yp[:,0]
				yp_val[val_indices] += yp_val_fold[:,0]
				yp_test[test_indices] += yp_test_fold[:,0]

			if extract_attention_weights:
				attention_weights[train_indices, :, :] += weight

	if predict:
	    yp_train /= ((num_folds - 2) * num_iterations)
	    yp_val /= num_iterations
	    yp_test /= num_iterations

	    if from_logits:
	    	yp_train = 1 / (1 + np.exp(-yp_train))
	    	yp_val = 1 / (1 + np.exp(-yp_val))
	    	yp_test = 1 / (1 + np.exp(-yp_test))

	    best_proba, metrics, test_pred = performance_analysis(meta_df, yp_train, yp_val, yp_test, predict_level=NN_level)

	if extract_attention_weights:
		attention_weights /= ((num_folds - 2) * num_iterations)


	return [yp_train, yp_val, yp_test], best_proba, metrics, test_pred, attention_weights


# =============================================================================
# ########################### Define Classifier & Training ####################
# =============================================================================
# In[] define the classifier & training
def training_classifier(X_data, y_data, feature_names, train_indices, val_indices, test_indices,
	classifier='LightGBM', verbose_eval=0, predict=True, pretrained=False, early_stopping_rounds=100,
	model_file_name='LightGBM_measurement_level_global_features_iter0_fold0.dat', units=(64,32)):

	train_X, train_y = X_data.values[train_indices], y_data[train_indices]
	val_X, val_y = X_data.values[val_indices], y_data[val_indices]
	test_X, test_y = X_data.values[test_indices], y_data[test_indices]

	if not pretrained:
		if classifier == 'random_forest':
		    class_weight = dict({0:0.5, 1:2.0})
		    model = RandomForestClassifier(bootstrap=True, 
		        class_weight=class_weight, criterion='gini',
		        max_depth=8, max_features='auto', max_leaf_nodes=None,
		        min_impurity_decrease=0.0, min_impurity_split=None,
		        min_samples_leaf=4, min_samples_split=10,
		        min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1,
		        oob_score=False,
		        random_state=23948, verbose=verbose_eval, warm_start=False)
		    model.fit(train_X, train_y)

		elif classifier == 'XGboost':
		    trn = xgb.DMatrix(train_X, label = train_y, feature_names=feature_names)
		    val = xgb.DMatrix(val_X, label = val_y, feature_names=feature_names)
		    test = xgb.DMatrix(test_X, label = test_y, feature_names=feature_names)
		    params = {'objective':'binary:logistic', 'nthread':4, 'eval_metric': 'logloss'}
		    evallist = [(trn, 'train'), (val, 'validation'), (test, 'test')]
		    model = xgb.train(params, trn, num_boost_round=10000, evals=evallist, 
		        verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds)

		elif classifier == 'LightGBM':
			params = {'objective': 'binary', 'boosting': 'gbdt', 'learning_rate': 0.01,
			    'num_leaves': 80, 'num_threads': 4, 'metric': 'binary_logloss',
			    'feature_fraction': 0.8, 'bagging_freq': 1, 'bagging_fraction': 0.8,
			    'seed': 23974, 'num_boost_round': 10000 }

			trn = lgb.Dataset(train_X, train_y, feature_name=feature_names)
			val = lgb.Dataset(val_X, val_y, feature_name=feature_names)
			test = lgb.Dataset(test_X, test_y, feature_name=feature_names)
		    # train model
			with warnings.catch_warnings():
			    warnings.simplefilter("ignore")
			    model = lgb.train(params, trn, valid_sets=(trn, test, val), 
			        valid_names=("train", "test", "validation"), fobj=None, feval=None,
			        early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)

		elif classifier == 'MLP':
			model = MLPClassifier(hidden_layer_sizes=units, random_state=1, verbose=verbose_eval)
			model.fit(train_X, train_y)

		elif classifier == 'Voting':
			# clf1 = LogisticRegression(random_state=1, max_iter=2000)
			clf2 = KNeighborsClassifier(n_neighbors=6)
			clf3 = GaussianNB()
			clf4 = SVC(kernel='rbf', probability=True)
			# clf5 = DecisionTreeClassifier(max_depth=8)
			# model = VotingClassifier(estimators=[('lr', clf1), ('knn', clf2), ('gnb', clf3),
			# 	('svc', clf4), ('dt', clf5)], voting='soft')
			model = VotingClassifier(estimators=[('knn', clf2), ('gnb', clf3), ('svc', clf4)],
				voting='soft')
			model = model.fit(train_X, train_y)

		pickle.dump(model, open(model_file_name, 'wb'))

	else:
		model = pickle.load(open(model_file_name, 'rb'))

	if predict:
		if classifier == 'random_forest' or classifier == 'MLP' or classifier == 'Voting':
		    yp = model.predict_proba(train_X)[:,1]
		    yp_val_fold = model.predict_proba(val_X)[:,1]
		    yp_test_fold = model.predict_proba(test_X)[:,1]

		elif classifier == 'XGboost':
		    yp = model.predict(xgb.DMatrix(train_X, feature_names=feature_names))
		    yp_val_fold = model.predict(xgb.DMatrix(val_X, feature_names=feature_names))
		    yp_test_fold = model.predict(xgb.DMatrix(test_X, feature_names=feature_names))

		elif classifier == 'LightGBM':
		    yp = model.predict(train_X)
		    yp_val_fold = model.predict(val_X)
		    yp_test_fold = model.predict(test_X)

	else:
		yp, yp_val_fold, yp_test_fold = None, None, None 


	return model, yp, yp_val_fold, yp_test_fold


# =============================================================================
# ######################## Whole Framework & Training #########################
# =============================================================================
def whole_process_training_single_iter(meta_df, signal_waves, global_features, 
	local_features=True, NN_level='signal', NN_model='LSTM', Dense_layers=2, NN_pretrained=True,
    layer_idx = 5, NN_batch_size = 512, output_folder='results_200chunks_weighted_bce', classifier='XGboost', 
    classifier_level='measurement', feature_set = 'global', kfold_random_state=123948, iter=0,
    early_stopping_rounds=100, num_folds=5, num_iterations=1, verbose_eval=0, load_local_features=True,
    predict=True, pretrained=False, monitor='val_loss', dropout=0.4, regularizer='l2', weights_dict=None,
    loss_name='weighted_bce', from_logits=True, kernel_size=[12,7], n_epochs=100, units=(128,64,32)):

	global_features, y_data, feature_names = combine_global_features(meta_df, global_features,
		classifier_level=classifier_level, feature_set=feature_set)
	if local_features:
		signal_y = meta_df['target'].values
		if load_local_features:
			all_local_features = pickle.load(open('{}/local_features_{}_{}Dense_layers_{}_level_layer_{}.dat'.format(output_folder, \
					NN_model, Dense_layers, NN_level, layer_idx),'rb'))

	if predict:
	    yp_train = np.zeros(global_features.shape[0])
	    yp_val = np.zeros(global_features.shape[0])
	    yp_test = np.zeros(global_features.shape[0])
	else:
	    yp_train = None
	    yp_val = None
	    yp_test = None
	    best_proba, metrics, test_pred = None, None, None 
     
	# models = []
	np.random.seed(kfold_random_state + iter)
	splits = np.zeros(global_features.shape[0], dtype=np.int)
	m = y_data == 1
	splits[m] = np.random.randint(0, 5, size=m.sum())
	m = y_data == 0
	splits[m] = np.random.randint(0, 5, size=m.sum())

	for fold in tqdm(range(num_folds)):
		print("Beginning iteration {}, fold {}".format(iter, fold))        	
		val_fold = fold
		test_fold = (fold + 1) % num_folds
		train_folds = [f for f in range(num_folds) if f not in [val_fold, test_fold]]

		train_indices = np.where(np.isin(splits, train_folds))[0]
		val_indices = np.where(splits == val_fold)[0]
		test_indices = np.where(splits == test_fold)[0]

		if local_features: 
			if load_local_features:
				inter_features = all_local_features[5*iter + fold]
			else:
				K.clear_session()
				print("NN Training & Extracting intermediate features")
				ckpt_name = '{}/RNN_weights/{}/{}_{}Dense_layers_{}_level_monitor_{}_iter{}_fold{}.h5'.format(output_folder, \
					loss_name, NN_model, Dense_layers, NN_level, monitor, iter, fold)
				_, _, _, inter_features, _ = Network_Training_single_fold(meta_df, signal_waves, 
					signal_y, train_indices, val_indices, test_indices, indice_level=classifier_level, 
					NN_level=NN_level, NN_model=NN_model, Dense_layers=Dense_layers, NN_pretrained=NN_pretrained,
					ckpt_name=ckpt_name, predict=False, intermediate_features=True, layer_idx=layer_idx, batch_size=NN_batch_size,
					monitor=monitor, dropout=dropout, regularizer=regularizer, loss_name=loss_name, from_logits=from_logits, 
					weights_dict=weights_dict, kernel_size=kernel_size, n_epochs=n_epochs)
			num_feature = inter_features.shape[1]
			#### Combine intermediate features & global features ####
			X_data, feature_names = combine_intermediate_features(global_features, \
				inter_features, NN_level=NN_level, classifier_level=classifier_level)

		else:
			X_data = global_features

		############# Input final features to the classifier ###################
		if not local_features:
		    model_file_name = '{}/models/global_scale/{}_{}_level_{}_features_iter{}_fold{}.dat'.format(output_folder, \
		    	classifier, classifier_level, feature_set, iter, fold)
		else:
		    model_file_name = '{}/models/{}_{}Dense_layers_{}_level_monitor_{}_{}interfeatures_{}_{}_level_iter{}_fold{}.dat'.format(output_folder, \
		    	NN_model, Dense_layers, NN_level, monitor, num_feature, classifier, classifier_level, iter, fold)

		print("Classifier Training: ")
		model, yp, yp_val_fold, yp_test_fold = training_classifier(X_data, y_data, feature_names, train_indices,
			val_indices, test_indices, classifier=classifier, predict=predict, pretrained=pretrained, 
			early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval, model_file_name=model_file_name,
			units=units)
		# models.append(model)

		if predict:
			yp_train[train_indices] += yp
			yp_val[val_indices] += yp_val_fold
			yp_test[test_indices] += yp_test_fold

	if predict:
	    yp_train /= ((num_folds - 2) * num_iterations)
	    yp_val /= num_iterations
	    yp_test /= num_iterations

	    best_proba, metrics, test_pred = performance_analysis(meta_df, yp_train, yp_val, yp_test, predict_level=classifier_level)

	return [yp_train, yp_val, yp_test], best_proba, metrics, test_pred


def whole_process_training(meta_df, signal_waves, global_features, 
	local_features=True, NN_level='signal', NN_model='LSTM', Dense_layers=2, NN_pretrained=True,
    layer_idx = 5, NN_batch_size = 512, output_folder='results_200chunks_weighted_bce', classifier='XGboost', 
    classifier_level='measurement', feature_set = 'global', kfold_random_state=123948,
    early_stopping_rounds=100, num_folds=5, num_iterations=25, verbose_eval=0, load_local_features=True,
    predict=True, pretrained=True, monitor='val_loss', dropout=0.4, weights_dict=None,
    regularizer='l2', loss_name='weighted_bce', from_logits=True, kernel_size=[12,7], n_epochs=100, 
    units=(64,32)):

	global_features, y_data, feature_names = combine_global_features(meta_df, global_features,
		classifier_level=classifier_level, feature_set=feature_set)
	if local_features:
		signal_y = meta_df['target'].values
		if load_local_features:
			all_local_features = pickle.load(open('{}/local_features_{}_{}Dense_layers_{}_level_layer_{}.dat'.format(output_folder, \
					NN_model, Dense_layers, NN_level, layer_idx),'rb'))
		else:
			all_local_features = []
		# 	all_local_features = np.zeros((global_features.shape[0], num_folds, num_iterations))

	if predict:
	    yp_train = np.zeros(global_features.shape[0])
	    yp_val = np.zeros(global_features.shape[0])
	    yp_test = np.zeros(global_features.shape[0])
	else:
	    yp_train = None
	    yp_val = None
	    yp_test = None
	    best_proba, metrics, test_pred = None, None, None 
     
	# models = []
	for iter in tqdm(range(num_iterations)):
		##### split the dataset 
		np.random.seed(kfold_random_state + iter)
		splits = np.zeros(global_features.shape[0], dtype=np.int)
		m = y_data == 1
		splits[m] = np.random.randint(0, 5, size=m.sum())
		m = y_data == 0
		splits[m] = np.random.randint(0, 5, size=m.sum())

		# for fold in tqdm(range(num_folds)):
		for fold in range(num_folds):
			# print("Beginning iteration {}, fold {}".format(iter, fold))        	
			val_fold = fold
			test_fold = (fold + 1) % num_folds
			train_folds = [f for f in range(num_folds) if f not in [val_fold, test_fold]]

			train_indices = np.where(np.isin(splits, train_folds))[0]
			val_indices = np.where(splits == val_fold)[0]
			test_indices = np.where(splits == test_fold)[0]

			if local_features: 
				if load_local_features:
					inter_features = all_local_features[5*iter + fold]
				else:
					K.clear_session()
					# print("NN Training & Extracting intermediate features")
					ckpt_name = '{}/RNN_weights/{}_{}Dense_layers_{}_level_monitor_{}_iter{}_fold{}.h5'.format(output_folder, \
						NN_model, Dense_layers, NN_level, monitor, iter, fold)
					_, _, _, inter_features, _ = Network_Training_single_fold(meta_df, signal_waves, 
						signal_y, train_indices, val_indices, test_indices, indice_level=classifier_level, 
						NN_level=NN_level, NN_model=NN_model, Dense_layers=Dense_layers, NN_pretrained=NN_pretrained,
						ckpt_name=ckpt_name, predict=False, intermediate_features=True, layer_idx=layer_idx, 
						batch_size=NN_batch_size, monitor=monitor, dropout=dropout, regularizer=regularizer, loss_name=loss_name,
						from_logits=from_logits, weights_dict=weights_dict, kernel_size=kernel_size, n_epochs=n_epochs)
					# num_feature = inter_features.shape[1]
					all_local_features.append(inter_features)

				#### Combine intermediate features & global features ####
				X_data, feature_names = combine_intermediate_features(global_features, \
					inter_features, NN_level=NN_level, classifier_level=classifier_level)
				num_feature = inter_features.shape[1]

			else:
				X_data = global_features

			############# Input final features to the classifier ###################
			if not local_features:
			    model_file_name = '{}/models/global_scale/{}_{}_level_{}_features_iter{}_fold{}.dat'.format(output_folder, \
			    	classifier, classifier_level, feature_set, iter, fold)
			else:
			    model_file_name = '{}/models/{}_{}Dense_layers_{}_level_monitor_{}_{}interfeatures_{}_{}_level_iter{}_fold{}.dat'.format(output_folder, \
			    	NN_model, Dense_layers, NN_level, monitor, num_feature, classifier, classifier_level, iter, fold)

			# print("Classifier Training: ")
			model, yp, yp_val_fold, yp_test_fold = training_classifier(X_data, y_data, 
                feature_names, train_indices, val_indices, test_indices, 
                classifier=classifier, predict=predict, pretrained=pretrained, 
				early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval, 
                model_file_name=model_file_name, units=units)
			# models.append(model)

			if predict:
				yp_train[train_indices] += yp
				yp_val[val_indices] += yp_val_fold
				yp_test[test_indices] += yp_test_fold

	if local_features:
		if not load_local_features:
			all_local_features = np.array(all_local_features)
			pickle.dump(all_local_features, open('{}/local_features_{}_{}Dense_layers_{}_level_layer_{}.dat'.format(output_folder, \
					NN_model, Dense_layers, NN_level, layer_idx), 'wb'))

	if predict:
	    yp_train /= ((num_folds - 2) * num_iterations)
	    yp_val /= num_iterations
	    yp_test /= num_iterations

	    best_proba, metrics, test_pred = performance_analysis(meta_df, yp_train, yp_val, yp_test, predict_level=classifier_level)

	return [yp_train, yp_val, yp_test], best_proba, metrics, test_pred
