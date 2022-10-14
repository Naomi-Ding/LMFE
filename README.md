# LMFE: Learning based Multi-scale Feature Engineering in Partial Discharge Detection


## Introduction
### Abstract
#### The partial discharge (PD) detection is of critical importance in the stability and continuity of power distribution operations. Although several feature engineering methods have been developed to refine and improve PD detection accuracy, they can be suboptimal due to several major issues: (i) failure in identifying fault-related pulses, (ii) the lack of inner-phase temporal representation, and (iii) multi-scale feature integration. The aim of this paper is to develop a Learning based Multiscale Feature Engineering (LMFE) framework for PD detection of each signal in a 3-phase power system, while addressing the above issues. The 3-phase measurements are first preprocessed to identify the pulses together with the surrounded waveforms. Next, our feature engineering is conducted to extract the global-scale features, i.e., phase level and measurement level aggregations of the pulse-level information, and the local-scale features focusing on waveforms and their inner-phase temporal information. A recurrent neural network (RNN) model is trained, and intermediate features are extracted from this trained RNN model. Furthermore, these multi-scale features are merged and fed into a classifier to distinguish the different patterns between faulty and non-faulty signals. Finally, our LMFE is evaluated by analyzing the VSB ENET dataset, which shows that LMFE outperforms existing approaches and provides the state-of-the-art solution in the PD detection.

<details>
  <summary>Examples of Three-phase Measurement Signals</summary>  
  
  ![Example1](/figures/Three-Phase_Measurement_normal.png)
  
  ![Example2](/figures/Three-Phase_Measurement_faulty.png)
  
  Fig 1. Examples of (a) normal signals and (b) faulty signals in 3-phase measurements.
  </details>


### Framework
![Framework](/figures/Framework.png)
Fig 2. The workflow of proposed LMFE for PD detection: (a) original signals for one measurement; (b) signals preprocessing including phase alignment,
signal flattening, and noise estimation; (c) pulses & surrounded waveforms identification; (d) multi-scale feature engineering including (1) global-scale features based on both phase-level and measurement-level summary statistics of features extracted from pulses and waveforms, and (2) local-scale features extracted from a RNN model using the sequential waveforms as input; (e) final detection based on combinations of multi-scale features.

<details>
	<summary>Details of the construction of local-scale features</summary>

	![RNN Model](/figures/RNN_model.png)

	Fig 3. Details of the construction of local-scale features: (a) the architecture of the RNN model; (b) the illustration of the bidirectional LSTM/GRU; (c) the illustration of feed-forward attention layer. Here T is the number of waveforms in each signal, w is the length of each waveform, and n is the number of signals.
	</details>

