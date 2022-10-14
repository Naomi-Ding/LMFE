import argparse 
parser = argparse.ArgumentParser(description='Choice of RNN model, classifier, training level and number of dense layers')
parser.add_argument('--data_dir', type=str, default='vsb-power-line-fault-detection/', help='the folder path for signal data')
parser.add_argument('--input_dir', type=str, default='processed_input/', help='the folder path for the preprocessed data')
parser.add_argument('--preprocessed', default=False, action='store_true', help='if the preprocessing steps have done for waveform and global feature extraction')
parser.add_argument('--recalculate_peaks', default=False, action='store_true', help='whether to recalculate the peaks in the preprocessing step, only valid when preprocessed=False')
parser.add_argument('--nchunks', type=int, default=160, help='number of chunks in waveforms, choose from 100,160,200,400')
parser.add_argument('--NN_level', type=str, default='signal', help='RNN training level')
parser.add_argument('--NN_model', type=str, default='LSTM', help='choose from LSTM and minimal_rnn, or TCN ')
parser.add_argument('--Dense_layers', type=int, default=2, help='number of dense lyaers in RNN, choose from 1,2,3')
parser.add_argument('--NN_pretrained', default=False, action='store_true', help='if the RNN model is well-trained')
parser.add_argument('--classifier', type=str, default='XGboost', help='classifier, choose from LightGBM, XGboost, random_forest')
parser.add_argument('--classifier_level', type=str, default='measurement', help='Classification training level')
parser.add_argument('--num_iterations', type=int, default=25, help='number of iterations for classifier')
parser.add_argument('--kfold_random_state', type=int, default=123948, help='random seeds for splitting the k-folds')
parser.add_argument('--num_folds', type=int, default=5, help='number of folds for cross-validation')
parser.add_argument('--feature_set', type=str, default='global', help='features to be input to classifier, choose from global, phase, measurement')
parser.add_argument('--local_features', default=False, action='store_true', help='include the intermediate features or not')
parser.add_argument('--load_local_features', default=False, action='store_true', help='whether to load the intermediate features or extract from the netwrok')
parser.add_argument('--iter', type=int, default=0, help='iteration index, ranging from 0 to num_iterations-1')
parser.add_argument('--layer_idx', type=int, default=5, help='layer_idx to extract the intermediate features')
parser.add_argument('--NN_batch_size', type=int, default=512, help='batch size for RNN training')
parser.add_argument('--monitor', type=str, default='val_loss', help='monitor the improvement of val_loss or val_matthews_correlation_2 (i.e., 1-mcc) in the RNN model')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout ratio in the RNN model')
parser.add_argument('--regularizer', type=str, default='l2', help='use l1 or l2 penalty in the RNN model')
parser.add_argument('--loss_name', type=str, default='weighted_bce', help='loss function used in the RNN model, choose from bce, weighted_bce, focal')
parser.add_argument('--from_logits', default=False, action='store_true', help='training from logits or not')
parser.add_argument('--kernel_size', default=[12,7], help='kernel size of Conv1D in TCN')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs for NN model')
parser.add_argument('--pretrained', default=False, action='store_true', help='if classifier is trained well')
parser.add_argument('--NN_only', default=False, action='store_true', help='if get results from the network only')
parser.add_argument('--units', default=(128,64,32), help='hidden layers in MLP classifier')
parser.add_argument('--predict', default=False, action='store_true', help='if the model is evaluated by prediction performance')
parser.add_argument('--load_attention_weights', default=False, action='store_true', help='whether to load the attention weights or recalculate it')
parser.add_argument('--extract_attention_weights', default=False, action='store_true', help='whether to extract the attention weights for analysis, only valid when load_attention_weights=False')
args = parser.parse_args()

print(args)

signal_len = 800000
data_dir = args.data_dir
input_dir = args.input_dir
preprocessed = args.preprocessed
recalculate_peaks = args.recalculate_peaks
nchunks = args.nchunks
window_size = int(signal_len / nchunks)
NN_level = args.NN_level
NN_model = args.NN_model
Dense_layers = args.Dense_layers
NN_pretrained = args.NN_pretrained
classifier = args.classifier
classifier_level = args.classifier_level
num_iterations = args.num_iterations
num_folds = args.num_folds
feature_set = args.feature_set
local_features = args.local_features
load_local_features = args.load_local_features
kfold_random_state = args.kfold_random_state
iter = args.iter 
layer_idx = args.layer_idx
NN_batch_size = args.NN_batch_size
monitor = args.monitor
dropout = args.dropout
regularizer = args.regularizer
loss_name = args.loss_name
from_logits = args.from_logits
kernel_size = args.kernel_size 
n_epochs = args.n_epochs 
pretrained = args.pretrained
NN_only = args.NN_only
units = args.units
predict = args.predict
load_attention_weights = args.load_attention_weights
extract_attention_weights = args.extract_attention_weights



# In[] # import libraries 
from utils_process import peaks_on_flatten, choose_chunk_peak, calculate_50hz_fourier_coefficient, process_measurement, create_global_features
import time 
import pickle
import pyarrow 
import pyarrow.parquet as pq
import pandas as pd
from utils_train import whole_process_training_single_iter, whole_process_training, whole_Network_training

# from sklearn.cluster import KMeans
# from matplotlib import pyplot as plt 

meta_df = pd.read_csv('metadata_train.csv')
signal_ids = meta_df['signal_id'].values
# input_dir = 'processed_input/'

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
	if not load_local_features:
		waveforms = pickle.load(open(input_dir + 'all_chunk_waves_{}chunks.dat'.format(nchunks), 'rb'))
	else:
		waveforms = None 

X_global.set_index('id_measurement', inplace=True)


# =============================================================================
# ###################### STEP 2 Model Training ################################
# =============================================================================
import random
import numpy as np
import tensorflow as tf
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

output_folder = 'results_{}chunks_{}'.format(nchunks, loss_name)

if num_iterations == 1:
	_, best_proba, metrics, test_pred = whole_process_training_single_iter(meta_df, waveforms, X_global,
			local_features=local_features, NN_level=NN_level, NN_model=NN_model,
			Dense_layers=Dense_layers, NN_pretrained=NN_pretrained, layer_idx=layer_idx, NN_batch_size=NN_batch_size, 
			output_folder=output_folder, classifier=classifier, classifier_level=classifier_level, num_folds=num_folds,
			num_iterations=num_iterations, feature_set=feature_set, kfold_random_state=kfold_random_state, iter=iter,
			pretrained=pretrained, load_local_features=load_local_features, predict=predict, early_stopping_rounds=100, 
			verbose_eval=0, weights_dict=weights_dict, monitor=monitor, dropout=dropout, regularizer=regularizer, 
			loss_name=loss_name, from_logits=from_logits, kernel_size=kernel_size, n_epochs=n_epochs, units=units)

else:
	if not NN_only:	
		_, best_proba, metrics, test_pred = whole_process_training(meta_df, waveforms, X_global,
				local_features=local_features, NN_level=NN_level, NN_model=NN_model,
				Dense_layers=Dense_layers, NN_pretrained=NN_pretrained, layer_idx=layer_idx, NN_batch_size=NN_batch_size, 
				output_folder=output_folder, classifier=classifier, classifier_level=classifier_level, num_folds=num_folds,
				num_iterations=num_iterations, feature_set=feature_set, kfold_random_state=kfold_random_state, 
				load_local_features=load_local_features, pretrained=pretrained, predict=predict, early_stopping_rounds=100, 
				verbose_eval=0, weights_dict=weights_dict, monitor=monitor, dropout=dropout, regularizer=regularizer, 
				loss_name=loss_name, from_logits=from_logits, kernel_size=kernel_size, n_epochs=n_epochs, units=units)
		test_pred.to_csv(output_folder + '/test_pred_{}_{}Dense_layers_{}_level_LAYER_{}_interfeatures_{}_{}_level.csv'.format(NN_model,
			Dense_layers, NN_level, layer_idx, classifier, classifier_level))

	else:
		_, best_proba_RNN, metrics_RNN, test_pred_RNN, attention_weights = whole_Network_training(meta_df, waveforms,
			NN_level=NN_level, NN_model=NN_model, Dense_layers=Dense_layers, NN_pretrained=NN_pretrained, 
			layer_idx=layer_idx, NN_batch_size=NN_batch_size, indice_level=classifier_level,
			output_folder=output_folder, kfold_random_state=kfold_random_state, num_folds=num_folds,
			num_iterations=num_iterations, predict=predict, monitor=monitor, dropout=dropout, regularizer=regularizer,
			kernel_size=kernel_size, from_logits=from_logits, loss_name=loss_name, 
			extract_attention_weights=extract_attention_weights)
		test_pred.to_csv(output_folder + '/test_pred_NNonly_{}_{}Dense_layers_{}_level.csv'.format(NN_model,
			Dense_layers, NN_level))
		if extract_attention_weights:
			pickle.dump(attention_weights, open(output_folder + '/attention_weights_{}_{}Dense_layers_{}_level.dat'.format(NN_model,
				Dense_layers, NN_level), 'wb'))
 