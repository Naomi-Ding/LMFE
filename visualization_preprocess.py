# =============================================================================
# ########################### Data Visualization ##############################
# =============================================================================
from scipy.signal import savgol_filter
from utils_process import get_crossing, phase_shift, noise_estimation_fixed, spike_detection_ori5_fast
import pyarrow 
import pyarrow.parquet as pq
import numpy as np 
from matplotlib import pyplot as plt 


data_dir = 'vsb-power-line-fault-detection/'
signal_df_0 = pq.read_pandas(data_dir + 'train.parquet', columns=[str(i) for i in range(3)]).to_pandas()
signal_df_1 = pq.read_pandas(data_dir + 'train.parquet', columns=[str(i) for i in range(3,6)]).to_pandas()
signal_len = signal_df_0.shape[0]

### Figure 1
x_time = np.linspace(0,20,signal_len)
plt.figure(figsize=(10,3))
plt.plot(x_time, signal_df_0)
plt.legend(['phase A', 'phase B', 'phase C'])
plt.xlim([0, 20])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (bit)')
plt.ylim([-60,60])
plt.title('Normal Signals')
plt.show()

plt.figure(figsize=(10,3))
plt.plot(x_time, signal_df_1)
plt.legend(['phase A', 'phase B', 'phase C'])
plt.xlim([0, 20])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (bit)')
plt.ylim([-60,60])
plt.title('Faulty Signals')
plt.show()

### Figure 2(b) Signal Pre-Processing
index = 0
signal = signal_df_0[str(index)].values
crossing = get_crossing(signal)
signal = phase_shift(signal, crossing)
yhat = savgol_filter(signal, 99, 3)
flat = signal - yhat
noise_level = noise_estimation_fixed(flat)
points = spike_detection_ori5_fast(flat, noise_level=noise_level)
points = np.array(points)

plt.plot(x_time, signal_df_0[str(index)])
plt.plot(x_time, signal)
plt.xlim([0,800000])
plt.title('Phase Alignment')
plt.xlim([0, 20])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (bit)')
plt.legend(['Original Signal', 'Aligned Signal'])
plt.show()

# plt.plot(signal_df_0[str(index)])
plt.plot(x_time, signal)
plt.plot(x_time, yhat)
plt.plot(x_time, flat)
plt.xlim([0, 20])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (bit)')
plt.title('Signal Flattening')
plt.legend(['Aligned Signal', 'Filtered Signal', 'Flattened Signal'])
plt.show()

plt.plot(x_time,flat)
plt.axhline(noise_level, color='r', linestyle='--')
plt.axhline(-noise_level, color='r', linestyle='--')
# plt.scatter(points[:,0], points[:,1], color='r')
plt.xlim([0, 20])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (bit)')
plt.title('Noise Level Estimation')
plt.show()

### Figure 2(c) Pulse Detection
fig, ax = plt.subplots(2,1, figsize=(10,6))
ax[0].plot(x_time, flat)
ax[0].axhline(noise_level, color='r', linestyle='--')
ax[0].axhline(-noise_level, color='r', linestyle='--')
ax[0].scatter(points[:,0]/800000*20, points[:,1], color='r')
ax[0].set_xlim([0,5])
# ax[0].set_xlabel('Time (ms)')
ax[0].set_ylabel('Amplitude (bit)')
# ax[0].set_title('Pulse Detection on Flattened Signal')

ax[1].plot(x_time-15, flat)
ax[1].axhline(noise_level, color='r', linestyle='--')
ax[1].axhline(-noise_level, color='r', linestyle='--')
ax[1].scatter(points[:,0]/800000*20-15, points[:,1], color='r')
ax[1].set_xlim([0,5])
ax[1].set_xlabel('Time (ms)')
ax[1].set_ylabel('Amplitude (bit)')
# ax[1].set_title('Pulse Detection on Flattened Signal')
plt.show()

