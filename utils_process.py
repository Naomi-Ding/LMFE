# In[] Import all the libraries 
import numpy as np
import pandas as pd
import pickle
import numba
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import gc
import os
import sys
import warnings
from numpy.fft import rfft, rfftfreq, irfft
from scipy.signal import savgol_filter
import pywt
from tqdm import tqdm

# =============================================================================
# ################## Utils for Preprocessing Waveforms ########################
# =============================================================================

def low_pass(s, threshold=1e4):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)
    
def get_crossing(x, phase=0):
    x_1 = low_pass(x, threshold=100)
    x = x_1.reshape((-1, ))
    zero_crossing = np.where(np.diff(np.sign(x)))[0]
    up_crossing = -1
    for zc in zero_crossing:
        if x[zc] < 0 and x[zc + 1] > 0:
            up_crossing = zc
    return up_crossing

def phase_shift(x, cross):
    if cross > 0:
        x = np.hstack([x[cross:], x[:cross]])
    return x

def noise_estimation_fixed(x):
    NUM_PATCH = 1000
    LEN_PATCH = 1000
    #indexes = np.random.uniform(low=0, high=799000, size=NUM_PATCH)
    indexes = np.linspace(0, 799000, NUM_PATCH)
    
    patches = np.zeros((NUM_PATCH, LEN_PATCH))
    for i in range(NUM_PATCH):
        patches[i, :] = x[int(indexes[i]): int(indexes[i]) + LEN_PATCH]
    coverage = 0
    diff = []
    over_th = 0
    for index, i in enumerate(np.linspace(0, 15, 31)):
        num_cover = np.sum(i > np.max(patches, axis=-1))
        diff.append(num_cover - coverage)
        
        if num_cover - coverage > 80:
            over_th = index
            
        coverage = num_cover
        
    loc_max_diff = np.argmax(diff)
    #print(diff)
    #plt.plot(diff)
    return max(loc_max_diff, over_th) * 0.5 + 0.5

def spike_detection_ori5_fast(x, size=250, noise_level=3):
    length = x.shape[0]
    x_abs = abs(x)
    
    NUM_PART = 20
    LEN_PART = int(length / NUM_PART)
    large_points = []
    for i in range(NUM_PART):
        large_points.append(np.argpartition(x_abs[i*LEN_PART: (i+1)*LEN_PART], 40000-100)[-100:] + i*LEN_PART)
    large_points = np.concatenate(large_points)
    
    x_clear = abs(x_abs)
    for point in large_points:
        if point - 50 > 0 and point + 50 < 800000:
            x_clear[point - 50: point + 50] = 0
    large_points_2 = []
    for i in range(NUM_PART):
        #large_points_2.append(np.argsort(x_clear[i*LEN_PART: (i+1)*LEN_PART])[-50:] + i*LEN_PART)
        #large_points_2.append(bottleneck.argpartition(-x_clear[i*LEN_PART: (i+1)*LEN_PART], 100)[:100] + i*LEN_PART)
        large_points_2.append(np.argpartition(x_clear[i*LEN_PART: (i+1)*LEN_PART], 40000-100)[-100:] + i*LEN_PART)
    large_points_2 = np.concatenate(large_points_2)
    large_points = np.concatenate([large_points, large_points_2])
    
    x_clear_2 = abs(x_clear)
    for point in large_points_2:
        if point - 50 > 0 and point + 50 < 800000:
            x_clear_2[point - 50: point + 50] = 0
    large_points_3 = []
    for i in range(NUM_PART):
        #large_points_3.append(np.argsort(x_clear_2[i*LEN_PART: (i+1)*LEN_PART])[-50:] + i*LEN_PART)
        #large_points_3.append(bottleneck.argpartition(-x_clear_2[i*LEN_PART: (i+1)*LEN_PART], 100)[:100] + i*LEN_PART)
        large_points_3.append(np.argpartition(x_clear_2[i*LEN_PART: (i+1)*LEN_PART], 40000-100)[-100:] + i*LEN_PART)
    large_points_3 = np.concatenate(large_points_3)
    large_points = np.concatenate([large_points, large_points_3])
    
    #large_points = np.argsort(x)[-2000:]
    points = []
    for point in large_points:
        if point - 25 > 0 and point + 25 < 800000:
            window = x_abs[max(0, point - 25): min(length, point + 25)]
            if x_abs[point] == np.max(window):
                if x_abs[point] > 4 and x_abs[point] < 50 and x_abs[point] > noise_level * 1.155:
                    
                    FLAG = True
                    while FLAG == True:
                        if np.sign(x[point-1]) * np.sign(x[point]) == -1 and x_abs[point-1] > 0.5 * x_abs[point]:
                            point = point - 1
                        else:
                            FLAG = False
                        
                    if np.sign(x[point-2]) * np.sign(x[point]) == -1 and x_abs[point-2] > 0.5 * x_abs[point]:
                        points.append([point-2, x[point]])  
                        
                    elif np.sign(x[point-3]) * np.sign(x[point]) == -1 and x_abs[point-3] > 0.5 * x_abs[point]:
                        points.append([point-3, x[point]]) 
                        
                    else:
                        points.append([point, x[point]])
    return points

def RMSE(x1, x2):
    rmse = np.sqrt(np.mean(np.power(x1 - x2, 2)))
    return rmse

def peaks_on_flatten(train_df, signal_ids, visualization=False):
    start_time = time.time()
    all_aligned_signals = []
    all_flat_signals = []
    all_points = []
    
    for index in signal_ids:
        if np.mod(index, 100) == 0:
            print(index)
            print('Elapsed time: {}'.format(time.time() - start_time))
            
        signal = train_df[str(index)].values
        
        crossing = get_crossing(signal)
        signal = phase_shift(signal, crossing)
        yhat = savgol_filter(signal, 99, 3)
        flat = signal - yhat
        noise_level = noise_estimation_fixed(flat)
        points = spike_detection_ori5_fast(flat, noise_level=noise_level)
        points = np.array(points)
        
        all_aligned_signals.append(signal)
        all_flat_signals.append(flat)
        all_points.append(points)

    if visualization:
        plt.plot(signal)
        plt.plot(yhat)
        plt.plot(flat)
        plt.axhline(noise_level, color='y')
        plt.axhline(-noise_level, color='y')
        plt.scatter(points[:,0], points[:,1], color='red')
        plt.legend(['aligned', 'high pass', 'flatten','noise'])
        plt.xlim([0,800000])
        plt.show()
    
    return all_aligned_signals, all_flat_signals, all_points

def choose_chunk_peak(all_flat_signals, all_points, window_size=5000, wave_len=15):
    num_window = all_flat_signals[0].shape[0] // window_size
    waves_all = np.zeros([len(all_flat_signals), num_window, 2*wave_len])
    
    start_time = time.time()
    for index in range(len(all_flat_signals)):
        if np.mod(index, 100) == 0:
            print(index)
            print('Elapsed time: {}'.format(time.time() - start_time))
        
        flat = all_flat_signals[index]
        points = all_points[index]
        if len(points) > 0:
            for i in range(num_window):
                flat_interval = flat[(i*window_size) : (i+1)*window_size]
                loc = (points[:,0] >= i*window_size) & (points[:,0] <= (i+1)*window_size)
                points_interval = points[loc]
                # points_interval = points[(points[:,0] >= i*window_size) & (points[:,0] <= (i+1)*window_size)]
                if len(points_interval) > 0:
                    point_keep = points_interval[np.argmax(np.abs(points_interval[:,1]))]
                    start = int(point_keep[0] - 15)
                    end = int(point_keep[0] + 15)
                    f = flat[start:end]
                    waves_all[index, i, :] = f
    
    return waves_all

# =============================================================================
# ################## Utils for Extracting Peaks #####################
# =============================================================================

@numba.jit(nopython=True)
def flatiron(x, alpha=100., beta=1):
    """
    Flatten signal
    Creator: Michael Kazachok
    Source: https://www.kaggle.com/miklgr500/flatiron
    """
    new_x = np.zeros_like(x)
    zero = x[0]
    for i in range(1, len(x)):
        zero = zero*(alpha-beta)/alpha + beta*x[i]/alpha
        new_x[i] = x[i] - zero
    return new_x

@numba.jit(nopython=True)
def drop_missing(intersect,sample):
    """
    Find intersection of sorted numpy arrays
    Since intersect1d sort arrays each time, it's effectively inefficient.
    Here you have to sweep intersection and each sample together to build
    the new intersection, which can be done in linear time, maintaining order. 

    Source: https://stackoverflow.com/questions/46572308/intersection-of-sorted-numpy-arrays
    Creator: B. M.
    """
    i=j=k=0
    new_intersect=np.empty_like(intersect)
    while i< intersect.size and j < sample.size:
        if intersect[i]==sample[j]: # the 99% case
            new_intersect[k]=intersect[i]
            k+=1
            i+=1
            j+=1
        elif intersect[i]<sample[j]:
            i+=1
        else : 
            j+=1
    return new_intersect[:k]

@numba.jit(nopython=True)
def _local_maxima_1d_window_single_pass(x, w):
    
    midpoints = np.empty(x.shape[0] // 2, dtype=np.intp)
    left_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    right_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    m = 0  # Pointer to the end of valid area in allocated arrays

    i = 1  # Pointer to current sample, first one can't be maxima
    i_max = x.shape[0] - 1  # Last sample can't be maxima
    while i < i_max:
        # Test if previous sample is smaller
        if x[i - 1] < x[i]:
            i_ahead = i + 1  # Index to look ahead of current sample

            # Find next sample that is unequal to x[i]
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1
                    
            i_right = i_ahead - 1
            
            f = False
            i_window_end = i_right + w
            while i_ahead < i_max and i_ahead < i_window_end:
                if x[i_ahead] > x[i]:
                    f = True
                    break
                i_ahead += 1
                
            # Maxima is found if next unequal sample is smaller than x[i]
            if x[i_ahead] < x[i]:
                left_edges[m] = i
                right_edges[m] = i_right
                midpoints[m] = (left_edges[m] + right_edges[m]) // 2
                m += 1
                
            # Skip samples that can't be maximum
            i = i_ahead - 1
        i += 1

    # Keep only valid part of array memory.
    midpoints = midpoints[:m]
    left_edges = left_edges[:m]
    right_edges = right_edges[:m]
    
    return midpoints, left_edges, right_edges

@numba.jit(nopython=True)
def local_maxima_1d_window(x, w=1):
    """
    Find local maxima in a 1D array.
    This function finds all local maxima in a 1D array and returns the indices
    for their midpoints (rounded down for even plateau sizes).
    It is a modified version of scipy.signal._peak_finding_utils._local_maxima_1d
    to include the use of a window to define how many points on each side to use in
    the test for a point being a local maxima.
    Parameters
    ----------
    x : ndarray
        The array to search for local maxima.
    w : np.int
        How many points on each side to use for the comparison to be True
    Returns
    -------
    midpoints : ndarray
        Indices of midpoints of local maxima in `x`.
    Notes
    -----
    - Compared to `argrelmax` this function is significantly faster and can
      detect maxima that are more than one sample wide. However this comes at
      the cost of being only applicable to 1D arrays.
    """    
        
    fm, fl, fr = _local_maxima_1d_window_single_pass(x, w)
    bm, bl, br = _local_maxima_1d_window_single_pass(x[::-1], w)
    bm = np.abs(bm - x.shape[0] + 1)[::-1]
    bl = np.abs(bl - x.shape[0] + 1)[::-1]
    br = np.abs(br - x.shape[0] + 1)[::-1]

    m = drop_missing(fm, bm)

    return m

@numba.jit(nopython=True)
def plateau_detection(grad, threshold, plateau_length=5):
    """Detect the point when the gradient has reach a plateau"""
    
    count = 0
    loc = 0
    for i in range(grad.shape[0]):
        if grad[i] > threshold:
            count += 1
        
        if count == plateau_length:
            loc = i - plateau_length
            break
            
    return loc

#@numba.jit(nopython=True)
def get_peaks(x, window=25,visualise=False,visualise_color=None):
    """
    Find the peaks in a signal trace.
    Parameters
    ----------
    x : ndarray
        The array to search.
    window : np.int
        How many points on each side to use for the local maxima test
    Returns
    -------
    peaks_x : ndarray
        Indices of midpoints of peaks in `x`.
    peaks_y : ndarray
        Absolute heights of peaks in `x`.
    x_hp : ndarray
        An absolute flattened version of `x`.
    """
    
    x_hp = flatiron(x, 100, 1)
    x_dn = np.abs(x_hp)

    peaks = local_maxima_1d_window(x_dn, window)
    heights = x_dn[peaks]
    ii = np.argsort(heights)[::-1]
    
    peaks = peaks[ii]
    heights = heights[ii]
    
    ky = heights
    kx = np.arange(1, heights.shape[0]+1)
    
    conv_length = 9

    grad = np.diff(ky, 1)/np.diff(kx, 1)
    grad = np.convolve(grad, np.ones(conv_length)/conv_length)#, mode='valid')
    grad = grad[conv_length-1:-conv_length+1]
    
    knee_x = plateau_detection(grad, -0.01, plateau_length=1000)
    knee_x -= conv_length//2
    
    if visualise:
        plt.plot(grad, color=visualise_color)
        plt.axvline(knee_x, ls="--", color=visualise_color)
    
    peaks_x = peaks[:knee_x]
    peaks_y = heights[:knee_x]
    
    ii = np.argsort(peaks_x)
    peaks_x = peaks_x[ii]
    peaks_y = peaks_y[ii]
        
    return peaks_x, peaks_y, x_hp

@numba.jit(nopython=True)
def clip(v, l, u):
    """Numba helper function to clip a value"""
    
    if v < l:
        v = l
    elif v > u:
        v = u
        
    return v

@numba.jit(nopython=True)
def create_sawtooth_template(sawtooth_length, pre_length, post_length):
    """Generate sawtooth template"""
    
    l = pre_length+post_length+1
    
    st = np.zeros(l)
    for i in range(sawtooth_length+1):
        
        j = pre_length+i
        if j < l:
            st[j] = 1 - ((2./sawtooth_length) * i)
        
    return st

@numba.jit(nopython=True)
def create_sawtooth_template1(pre_length, post_length):
    """Generate sawtooth template"""
    
    l = pre_length+post_length+1
    
    st = np.zeros(l)
    start = pre_length
    st[start] = 1
    st[start + 1] = -1
    st[start + 2] = 1
        
    return st

@numba.jit(nopython=True)
def create_sawtooth_template2(pre_length, post_length):
    """Generate sawtooth template"""
    
    l = pre_length+post_length+1
    
    st = np.zeros(l)
    start = pre_length
    st[start] = 1
    st[start + 1] = -1
    st[start + 2] = 0
    st[start + 3] = 0.5
        
    return st

@numba.jit(nopython=True)    
def create_sawtooth_template3(pre_length, post_length):
    """Generate sawtooth template"""
    
    l = pre_length+post_length+1
    
    st = np.zeros(l)
    start = pre_length
    st[start] = 1
    st[start + 1] = 0
    st[start + 2] = -1
        
    return st

@numba.jit(nopython=True)    
def create_sawtooth_template4(pre_length, post_length):
    """Generate sawtooth template"""
    
    l = pre_length+post_length+1
    
    st = np.zeros(l)
    start = pre_length
    st[start] = 1
        
    return st

@numba.jit(nopython=True)    
def create_sawtooth_template5(pre_length, post_length):
    """Generate sawtooth template"""
    
    l = pre_length+post_length+1
    
    st = np.zeros(l)
    start = pre_length
    st[start] = 1
    st[start + 1] = -1
    st[start + 2] = 1
    st[start + 3] = -1
        
    return st

@numba.jit(nopython=True) 
def create_sawtooth_template6(pre_length, post_length):
    """Generate sawtooth template"""
    
    l = pre_length+post_length+1
    
    st = np.zeros(l)
    start = pre_length
    st[start] = 1
    st[start + 1] = 0
    st[start + 2] = -1
    st[start + 3] = 1
        
    return st

@numba.jit(nopython=True)     
def create_sawtooth_template7(pre_length, post_length):
    """Generate sawtooth template"""
    
    l = pre_length+post_length+1
    
    st = np.zeros(l)
    start = pre_length
    st[start] = 1
    st[start + 1] = -1
        
    return st

@numba.jit(nopython=True)
def calculate_peak_features(px, x_hp0, ws=5, wl=25):
    """
    Calculate features for peaks.
    Parameters
    ----------
    px : ndarray
        Indices of peaks.
    x_hp0 : ndarray
        The array to search.
    ws : np.int
        How many points on each side to use for small window features
    wl : np.int
        How many points on each side to use for large window features
    Returns
    -------
    features : ndarray
        Features calculate for each peak in `x_hp0`.
    """
    peak_features_names = [
        'ratio_next',
        'ratio_prev',
        'small_dist_to_min',
        'sawtooth_rmse',
        'sawtooth_rmse1',
        'sawtooth_rmse2',
        'sawtooth_rmse3',
        'sawtooth_rmse4',
        'sawtooth_rmse5',
        'sawtooth_rmse6',
        'sawtooth_rmse7',
    ]
    num_peak_features = len(peak_features_names)
    features = np.ones((px.shape[0], num_peak_features), dtype=np.float64) * np.nan
    
    for i in range(px.shape[0]):
        
        feature_number = 0
        
        x = px[i]
        x_next = x+1
        x_prev = x-1
        
        h0 = x_hp0[x]

        ws_s = clip(x-ws, 0, 800000-1)
        ws_e = clip(x+ws, 0, 800000-1)
        wl_s = clip(x-wl, 0, 800000-1)
        wl_e = clip(x+wl, 0, 800000-1)
        
        ws_pre = x - ws_s
        ws_post = ws_e - x
        
        wl_pre = x - wl_s
        wl_post = wl_e - x
        
        if x_next < 800000:
            h0_next = x_hp0[x_next]
            features[i, feature_number] = np.abs(h0_next)/np.abs(h0)
        feature_number += 1
            
        if x_prev >= 0:
            h0_prev = x_hp0[x_prev]
            features[i, feature_number] = np.abs(h0_prev)/np.abs(h0)
        feature_number += 1
            
        x_hp_ws0 = x_hp0[ws_s:ws_e+1]
        x_hp_wl0 = x_hp0[wl_s:wl_e+1]
        x_hp_wl0_norm = (x_hp_wl0/np.abs(h0))
        x_hp_ws0_norm = (x_hp_ws0/np.abs(h0))
        x_hp_abs_wl0 = np.abs(x_hp_wl0)
        wl_max_0 = np.max(x_hp_abs_wl0)
        
        ws_opp_peak_i = np.argmin(x_hp_ws0*np.sign(h0))
        
        features[i, feature_number] = ws_opp_peak_i - ws
        feature_number += 1
        
        x_hp_wl0_norm_sign = x_hp_wl0_norm * np.sign(h0)
        
        sawtooth_length = 3
        st = create_sawtooth_template(sawtooth_length, wl_pre, wl_post)
        assert np.argmax(st) == np.argmax(x_hp_wl0_norm_sign)
        assert st.shape[0] == x_hp_wl0_norm_sign.shape[0]
        features[i, feature_number] = np.mean(np.power(x_hp_wl0_norm_sign - st, 2))
        feature_number += 1

        st = create_sawtooth_template1(wl_pre, wl_post)
        features[i, feature_number] = np.mean(np.power(x_hp_wl0_norm_sign - st, 2))
        feature_number += 1

        st = create_sawtooth_template2(wl_pre, wl_post)
        features[i, feature_number] = np.mean(np.power(x_hp_wl0_norm_sign - st, 2))
        feature_number += 1

        st = create_sawtooth_template3(wl_pre, wl_post)
        features[i, feature_number] = np.mean(np.power(x_hp_wl0_norm_sign - st, 2))
        feature_number += 1

        st = create_sawtooth_template4(wl_pre, wl_post)
        features[i, feature_number] = np.mean(np.power(x_hp_wl0_norm_sign - st, 2))
        feature_number += 1

        st = create_sawtooth_template5(wl_pre, wl_post)
        features[i, feature_number] = np.mean(np.power(x_hp_wl0_norm_sign - st, 2))
        feature_number += 1    
        
        st = create_sawtooth_template6(wl_pre, wl_post)
        features[i, feature_number] = np.mean(np.power(x_hp_wl0_norm_sign - st, 2))
        feature_number += 1    
        
        st = create_sawtooth_template7(wl_pre, wl_post)
        features[i, feature_number] = np.mean(np.power(x_hp_wl0_norm_sign - st, 2))
        feature_number += 1    
        
        if i == 0:
            assert feature_number == num_peak_features
        
    return features

def process_signal(data,window=25):
    """
    Process a signal trace to find the peaks and calculate features for each peak.
    Parameters
    ----------
    data : ndarray
        The array to search.
    window : np.int
        How many points on each side to use for the local maxima test
    Returns
    -------
    px0 : ndarray
        Indices for each peak in `data`.
    height0 : ndarray
        Absolute heaight for each peak in `data`.
    f0 : ndarray
        Features calculate for each peak in `data`.
    """
    
    px0, height0, x_hp0 = get_peaks(
        data.astype(np.float),
        window=window, 
    )
            
    f0 = calculate_peak_features(px0, x_hp0)
    
    return px0, height0, f0

def process_measurement_peaks(data, signal_ids):
    """
    Process three signal traces in measurment to find the peaks
    and calculate features for each peak.
    Parameters
    ----------
    data : ndarray
        Signal traces.
    signal_ids : ndarray
        Signal IDs for each of the signal traces in measurment
    Returns
    -------
    res : ndarray
        Data for each peak in the three traces in `data`.
    sigid_res : ndarray
        Signal ID for each row in `res`.
    """
    res = []
    sigid_res = []
    
    assert data.shape[1] % 3 == 0
    N = data.shape[1]//3
    
    for i in range(N):
        
        sigids = signal_ids[i*3:(i+1)*3]
        x = data[:, i*3:(i+1)*3].astype(np.float)
        
        px0, height0, f0 = process_signal(x[:, 0])
        px1, height1, f1 = process_signal(x[:, 1])
        px2, height2, f2 = process_signal(x[:, 2])
        
        if px0.shape[0] != 0:
            res.append(np.hstack([
                px0[:, np.newaxis], 
                height0[:, np.newaxis],
                f0,
            ]))
            
            sigid_res.append(np.ones(px0.shape[0], dtype=np.int) * sigids[0])
        
        if px1.shape[0] != 0:
            res.append(np.hstack([
                px1[:, np.newaxis], 
                height1[:, np.newaxis],
                f1,
            ]))

            sigid_res.append(np.ones(px1.shape[0], dtype=np.int) * sigids[1])
        
        if px2.shape[0] != 0:
            res.append(np.hstack([
                px2[:, np.newaxis], 
                height2[:, np.newaxis],
                f2,
            ]))

            sigid_res.append(np.ones(px2.shape[0], dtype=np.int) * sigids[2])
            
    return res, sigid_res

def process_measurement(data_array, meta_df, fft_data):
    """
    Process three signal traces in measurment to find the peaks
    and calculate features for each peak.
    Parameters
    ----------
    # data_df : pandas.DataFrame
    #     Signal traces.
    data_array : ndarray
        Signal traces.
    meta_df : pandas.DataFrame
        Meta data for measurement
    fft_data : ndarray
        50Hz fourier coefficient for three traces
    Returns
    -------
    peaks : pandas.DataFrame
        Data for each peak in the three traces in `data`.
    """
    peaks, sigids = process_measurement_peaks(
        # data_df.values, # [:, :100*3], 
        data_array,
        meta_df['signal_id'].values, # [:100*3]
    )
    
    peaks = np.concatenate(peaks)

    peaks = pd.DataFrame(
        peaks,
        columns=['px', 'height'] + peak_features_names
    )
    peaks['signal_id'] = np.concatenate(sigids)

    # Calculate the phase resolved location of each peak
    phase_50hz = np.angle(fft_data, deg=False) # fft_data[:, 1]
    phase_50hz = pd.DataFrame(
        phase_50hz,
        columns=['phase_50hz']
    )
    phase_50hz['signal_id'] = meta_df['signal_id'].values

    peaks = pd.merge(peaks, phase_50hz, on='signal_id', how='left')

    dt = (20e-3/(800000))
    f1 = 50
    w1 = 2*np.pi*f1
    peaks['phase_aligned_x'] = (np.degrees(
        (w1*peaks['px'].values*dt) + peaks['phase_50hz'].values
    ) + 90) % 360
    
    # Calculate the phase resolved quarter for each peak
    peaks['Q'] = pd.cut(peaks['phase_aligned_x'], [0, 90, 180, 270, 360], labels=[0, 1, 2, 3])
    
    return peaks

@numba.jit(nopython=True, parallel=True)
def calculate_50hz_fourier_coefficient(data):
    """Calculate the 50Hz Fourier coefficient of a signal.
    Assumes the signal is 800000 data points long and covering 20ms.
    """

    n = 800000
    assert data.shape[0] == n
    
    omegas = np.exp(-2j * np.pi * np.arange(n) / n).reshape(n, 1)
    m_ = omegas ** np.arange(1, 2)
    
    m = m_.flatten()

    res = np.zeros(data.shape[1], dtype=m.dtype)
    for i in numba.prange(data.shape[1]):
        res[i] = m.dot(data[:, i].astype(m.dtype))
            
    return res

def process(peaks_df, meta_df):

    results = pd.DataFrame(index=meta_df['id_measurement'].unique())
    results.index.rename('id_measurement', inplace=True)
    
    ################################################################################
    
    if not USE_SIMPLIFIED_VERSION:
        # Filter peaks using ratio_next and height features
        # Note: may not be all that important
        peaks_df = peaks_df[~(
            (peaks_df['ratio_next'] > 0.33333)
            & (peaks_df['height'] > 50)
        )]
    
    ################################################################################

    # Count peaks in phase resolved quarters 0 and 2
    p = peaks_df[peaks_df['Q'].isin([0, 2])].copy()
    res = p.groupby('id_measurement').agg(
    {
        'px': ['count'],
    })
    res.columns = ["peak_count_Q02"]
    results = pd.merge(results, res, on='id_measurement', how='left')
        
    ################################################################################
    
    # Count total peaks for each measurement id
    res = peaks_df.groupby('id_measurement').agg(
    {
        'px': ['count'],
    })
    res.columns = ["peak_count_total"]
    results = pd.merge(results, res, on='id_measurement', how='left')

    ################################################################################

    # Count peaks in phase resolved quarters 1 and 3
    p = peaks_df[peaks_df['Q'].isin([1, 3])].copy()
    res = p.groupby('id_measurement').agg(
    {
        'px': ['count'],
    })
    res.columns = ['peak_count_Q13']
    results = pd.merge(results, res, on='id_measurement', how='left')
    
    ################################################################################
    
    # Calculate additional features using phase resolved quarters 0 and 2
    
    feature_quarters = [0, 2]
    
    p = peaks_df[peaks_df['Q'].isin(feature_quarters)].copy()
    
    p['abs_small_dist_to_min'] = np.abs(p['small_dist_to_min'])
    
    res = p.groupby('id_measurement').agg(
    {
        
        'height': ['mean', 'std'],
        'ratio_prev': ['mean'],
        'ratio_next': ['mean'],
        'abs_small_dist_to_min': ['mean'],        
        'sawtooth_rmse': ['mean'],
        'sawtooth_rmse1': ['mean'],
        'sawtooth_rmse2': ['mean'],
        'sawtooth_rmse3': ['mean'],
        'sawtooth_rmse4': ['mean'],
        'sawtooth_rmse5': ['mean'],
        'sawtooth_rmse6': ['mean'],
        'sawtooth_rmse7': ['mean'],
    })
    res.columns = ["_".join(f) + '_Q02' for f in res.columns]     
    results = pd.merge(results, res, on='id_measurement', how='left')
        
    return results


def get_features(signal_ids, kmeans, kmeans_A, kmeans_B, kmeans_C):
    signal_to_peak_dict = {}
    waves_all = []
    waves_all_unnorm = []
    point_count = 0

    signal_to_peak_dict_A = {}
    waves_all_A = []
    waves_all_unnorm_A = []
    point_count_A = 0
    
    signal_to_peak_dict_B = {}
    waves_all_B = []
    waves_all_unnorm_B = []
    point_count_B = 0
    
    signal_to_peak_dict_C = {}
    waves_all_C = []
    waves_all_unnorm_C = []
    point_count_C = 0
    
    num_types_all = 15
    num_types_sep = 6
    NUM_SEGS = 20
    NUM_SIGNALS = len(signal_ids)
    
    hist_types = np.zeros((NUM_SIGNALS, num_types_all, NUM_SEGS))
    hist_types_A = np.zeros((NUM_SIGNALS // 3, num_types_sep, NUM_SEGS))
    hist_types_B = np.zeros((NUM_SIGNALS // 3, num_types_sep, NUM_SEGS))
    hist_types_C = np.zeros((NUM_SIGNALS // 3, num_types_sep, NUM_SEGS))
    
    for index in signal_ids:
        if np.mod(index, 100) == 0:
            print(index)
            
        # signal = praq_train[str(index)].values
        signal = pq.read_pandas('train.parquet', columns = str(index)).to_pandas().values.reshape(-1)
        
        crossing = get_crossing(signal)
        signal = phase_shift(signal, crossing)
        yhat = savgol_filter(signal, 99, 3)
        flat = signal - yhat
        noise_level = noise_estimation_fixed(flat)
        points = spike_detection_ori5_fast(flat, noise_level=noise_level)
        points = np.array(points)
    
        if len(points) > 0:

            for point in points:
                #peak_dict[point_count] = (index, point, meta_train.iloc[index].target)
                
                start = int(point[0] - 15)
                end = int(point[0] + 15)        
                f = flat[start: end]
                f = f * np.sign(f[15])
                waves_all_unnorm.append(f)
                f = f / np.max(abs(f))
                waves_all.append(f)
                
                if index in signal_to_peak_dict.keys():
                    signal_to_peak_dict[index].append(point_count)
                else:
                    signal_to_peak_dict[index] = [point_count]
                
                point_count += 1
                
                pred = kmeans.predict(f[None, :])
                pos = point[0]
                pos_id = int(pos // 40000)
                hist_types[index, pred, pos_id] += 1

            if np.mod(index, 3) == 0:
                for point in points:
                    start = int(point[0] - 15)
                    end = int(point[0] + 15)        
                    f = flat[start: end]
                    f = f * np.sign(f[15])
                    waves_all_unnorm_A.append(f)
                    f = f / np.max(abs(f))
                    waves_all_A.append(f)
                    
                    if index in signal_to_peak_dict_A.keys():
                        signal_to_peak_dict_A[index].append(point_count_A)
                    else:
                        signal_to_peak_dict_A[index] = [point_count_A]
                    
                    point_count_A += 1
                    
                    pred = kmeans_A.predict(f[None, :])
                    pos = point[0]
                    pos_id = int(pos // 40000)
                    hist_types_A[index // 3, pred, pos_id] += 1
                
            elif np.mod(index, 3) == 1:
                for point in points:
                    start = int(point[0] - 15)
                    end = int(point[0] + 15)        
                    f = flat[start: end]
                    f = f * np.sign(f[15])
                    waves_all_unnorm_B.append(f)
                    f = f / np.max(abs(f))
                    waves_all_B.append(f)
                    
                    if index in signal_to_peak_dict_B.keys():
                        signal_to_peak_dict_B[index].append(point_count_B)
                    else:
                        signal_to_peak_dict_B[index] = [point_count_B]
                    
                    point_count_B += 1
                    
                    pred = kmeans_B.predict(f[None, :])
                    pos = point[0]
                    pos_id = int(pos // 40000)
                    hist_types_B[index // 3, pred, pos_id] += 1
                    
            else:
                for point in points:
                    start = int(point[0] - 15)
                    end = int(point[0] + 15)        
                    f = flat[start: end]
                    f = f * np.sign(f[15])
                    waves_all_unnorm_C.append(f)
                    f = f / np.max(abs(f))
                    waves_all_C.append(f)
                    
                    if index in signal_to_peak_dict_C.keys():
                        signal_to_peak_dict_C[index].append(point_count_C)
                    else:
                        signal_to_peak_dict_C[index] = [point_count_C]
                    
                    point_count_C += 1
                    
                    pred = kmeans_C.predict(f[None, :])
                    pos = point[0]
                    pos_id = int(pos // 40000)
                    hist_types_C[index // 3, pred, pos_id] += 1  

    num_types_all = 15
    x = np.array(waves_all)
    y_pred = kmeans.predict(x)

    num_types_sep = 6
    x_A = np.array(waves_all_A)
    y_pred_A = kmeans_A.predict(x_A)
    
    x_B = np.array(waves_all_B)
    y_pred_B = kmeans_B.predict(x_B)
    
    x_C = np.array(waves_all_C)
    y_pred_C = kmeans_C.predict(x_C)
    
    num_waveforms = int(len(signal_ids) / 3)
    
    number_peaks = np.zeros((num_waveforms, num_types_all))
    mean_height_peaks = np.zeros((num_waveforms, num_types_all))
    std_height_peaks = np.zeros((num_waveforms, num_types_all))
    mean_RMSE_peaks = np.zeros((num_waveforms, num_types_all))

    number_peaks_A = np.zeros((num_waveforms, num_types_sep))
    mean_height_peaks_A = np.zeros((num_waveforms, num_types_sep))
    std_height_peaks_A = np.zeros((num_waveforms, num_types_sep))
    mean_RMSE_peaks_A = np.zeros((num_waveforms, num_types_sep))
    
    number_peaks_B = np.zeros((num_waveforms, num_types_sep))
    mean_height_peaks_B = np.zeros((num_waveforms, num_types_sep))
    std_height_peaks_B = np.zeros((num_waveforms, num_types_sep))
    mean_RMSE_peaks_B = np.zeros((num_waveforms, num_types_sep))
    
    number_peaks_C = np.zeros((num_waveforms, num_types_sep))
    mean_height_peaks_C = np.zeros((num_waveforms, num_types_sep))
    std_height_peaks_C = np.zeros((num_waveforms, num_types_sep))
    mean_RMSE_peaks_C = np.zeros((num_waveforms, num_types_sep))

    combined = np.zeros((num_waveforms, 1))
    A = np.zeros((num_waveforms, 1))
    B = np.zeros((num_waveforms, 1))
    C = np.zeros((num_waveforms, 1))
    
    templates = np.zeros((num_types_all, 30))
    for i in range(num_types_all):
        templates[i] = np.mean(x[y_pred == i], axis=0)

    templates_A = np.zeros((num_types_sep, 30))
    for i in range(num_types_sep):
        templates_A[i] = np.mean(x_A[y_pred_A == i], axis=0)

    templates_B = np.zeros((num_types_sep, 30))
    for i in range(num_types_sep):
        templates_B[i] = np.mean(x_B[y_pred_B == i], axis=0)
        
    templates_C = np.zeros((num_types_sep, 30))
    for i in range(num_types_sep):
        templates_C[i] = np.mean(x_C[y_pred_C == i], axis=0)
        
    for ii in range(num_waveforms):
        
        heights = []
        RMSEs = []
        heights_A, heights_B, heights_C = [], [], []
        RMSEs_A, RMSEs_B, RMSEs_C = [], [], []

        #######################################################################
        # new feature
        combined[ii] = np.sum(hist_types[ii*3: ii*3+3, 1,  18: 20]) + \
                       np.sum(hist_types[ii*3: ii*3+3, 3,  9: 14]) + \
                       np.sum(hist_types[ii*3: ii*3+3, 5,  11: 14]) + \
                       np.sum(hist_types[ii*3: ii*3+3, 13, 4: 7]) + np.sum(hist_types[ii*3: ii*3+3, 13, 16: 18]) + \
                       np.sum(hist_types[ii*3: ii*3+3, 14, 0]) + np.sum(hist_types[ii*3: ii*3+3, 14, 17: 20])
                       
        #######################################################################
        
        for _ in range(num_types_all):
            heights.append([])
            RMSEs.append([])
            
        for _ in range(num_types_sep):
            heights_A.append([])
            RMSEs_A.append([])
            heights_B.append([])
            RMSEs_B.append([])
            heights_C.append([])
            RMSEs_C.append([])
            
        for i in [ii*3, ii*3+1, ii*3+2]:
            if i not in signal_to_peak_dict.keys():
                continue
            peak_indexes = signal_to_peak_dict[i]
            
            for index in peak_indexes:
                number_peaks[ii, y_pred[index]] += 1
                heights[y_pred[index]].append(waves_all_unnorm[index][15])
                #for j in range(num_types):
                #    RMSEs[j].append(RMSE(waves_all[index], templates[j]))
                RMSEs[y_pred[index]].append(RMSE(waves_all[index], templates[y_pred[index]]))

        for i in [ii*3, ii*3+1, ii*3+2]:
            if i in signal_to_peak_dict_A.keys():
                peak_indexes = signal_to_peak_dict_A[i]
                
                for index in peak_indexes:
                    number_peaks_A[ii, y_pred_A[index]] += 1
                    heights_A[y_pred_A[index]].append(waves_all_unnorm_A[index][15])
                    #for j in range(num_types):
                    #    RMSEs_A[j].append(RMSE(waves_all_A[index], templates_A[j]))
                    RMSEs_A[y_pred_A[index]].append(RMSE(waves_all_A[index], templates_A[y_pred_A[index]]))

                ###############################################################
                # new feature
                A[ii] = np.sum(hist_types_A[ii, 1, 0]) + np.sum(hist_types_A[ii, 1, 8: 11]) + np.sum(hist_types_A[ii, 1, 16: 20]) + \
                        np.sum(hist_types_A[ii, 3, 1: 3]) + \
                        np.sum(hist_types_A[ii, 4, 0]) + np.sum(hist_types_A[ii, 4, 8: 11]) + np.sum(hist_types_A[ii, 4, 16: 20]) + \
                        np.sum(hist_types_A[ii, 5, 18: 20])
                ###############################################################
                    
            if i in signal_to_peak_dict_B.keys():
                peak_indexes = signal_to_peak_dict_B[i]
                
                for index in peak_indexes:
                    number_peaks_B[ii, y_pred_B[index]] += 1
                    heights_B[y_pred_B[index]].append(waves_all_unnorm_B[index][15])
                    #for j in range(num_types):
                    #    RMSEs_B[j].append(RMSE(waves_all_B[index], templates_B[j]))
                    RMSEs_B[y_pred_B[index]].append(RMSE(waves_all_B[index], templates_B[y_pred_B[index]]))

                ###############################################################
                # new feature
                B[ii] = np.sum(hist_types_B[ii, 0, 17: 20]) + \
                        np.sum(hist_types_B[ii, 1, 9: 15]) + np.sum(hist_types_B[ii, 1, 19]) + \
                        np.sum(hist_types_B[ii, 2, 2: 5]) + np.sum(hist_types_B[ii, 2, 11: 15]) + \
                        np.sum(hist_types_B[ii, 3, 7: 10]) + \
                        np.sum(hist_types_B[ii, 4, 0:]) + np.sum(hist_types_B[ii, 4, 15]) + \
                        np.sum(hist_types_B[ii, 5, 10: 14])
                ###############################################################
                    
            if i in signal_to_peak_dict_C.keys():
                peak_indexes = signal_to_peak_dict_C[i]
                
                for index in peak_indexes:
                    number_peaks_C[ii, y_pred_C[index]] += 1
                    heights_C[y_pred_C[index]].append(waves_all_unnorm_C[index][15])
                    #for j in range(num_types):
                    #    RMSEs_C[j].append(RMSE(waves_all_C[index], templates_C[j]))
                    RMSEs_C[y_pred_C[index]].append(RMSE(waves_all_C[index], templates_C[y_pred_C[index]]))

                ###############################################################
                # new feature
                C[ii] = np.sum(hist_types_C[ii, 1, 5: 7]) + np.sum(hist_types_C[ii, 1, 15: 17]) + \
                        np.sum(hist_types_C[ii, 2, 3: 6]) + \
                        np.sum(hist_types_C[ii, 3, 11: 13]) + \
                        np.sum(hist_types_C[ii, 4, 4: 8]) + \
                        np.sum(hist_types_C[ii, 5, 1: 3]) + np.sum(hist_types_C[ii, 5, 11: 13])
                ###############################################################
                    
        for j in range(num_types_all):
            if len(heights[j]) > 0:
                mean_height_peaks[ii, j] = np.mean(heights[j])
                std_height_peaks[ii, j] = np.std(heights[j])
                mean_RMSE_peaks[ii, j] = np.mean(RMSEs[j])
            else:
                mean_height_peaks[ii, j] = -1
                std_height_peaks[ii, j] = -1
                mean_RMSE_peaks[ii, j] = -1  

        for j in range(num_types_sep):
            if len(heights_A[j]) > 0:
                mean_height_peaks_A[ii, j] = np.mean(heights_A[j])
                std_height_peaks_A[ii, j] = np.std(heights_A[j])
                mean_RMSE_peaks_A[ii, j] = np.mean(RMSEs_A[j])
            else:
                mean_height_peaks_A[ii, j] = -1
                std_height_peaks_A[ii, j] = -1
                mean_RMSE_peaks_A[ii, j] = -1
                
            if len(heights_B[j]) > 0:
                mean_height_peaks_B[ii, j] = np.mean(heights_B[j])
                std_height_peaks_B[ii, j] = np.std(heights_B[j])
                mean_RMSE_peaks_B[ii, j] = np.mean(RMSEs_B[j])
            else:
                mean_height_peaks_B[ii, j] = -1
                std_height_peaks_B[ii, j] = -1
                mean_RMSE_peaks_B[ii, j] = -1 
                
            if len(heights_C[j]) > 0:
                mean_height_peaks_C[ii, j] = np.mean(heights_C[j])
                std_height_peaks_C[ii, j] = np.std(heights_C[j])
                mean_RMSE_peaks_C[ii, j] = np.mean(RMSEs_C[j])
            else:
                mean_height_peaks_C[ii, j] = -1
                std_height_peaks_C[ii, j] = -1
                mean_RMSE_peaks_C[ii, j] = -1
    
    type_features = [combined, A, B, C]
    return number_peaks, mean_height_peaks, std_height_peaks, mean_RMSE_peaks, templates, number_peaks_A, mean_height_peaks_A, std_height_peaks_A, mean_RMSE_peaks_A, templates_A, number_peaks_B, mean_height_peaks_B, std_height_peaks_B, mean_RMSE_peaks_B, templates_B, number_peaks_C, mean_height_peaks_C, std_height_peaks_C, mean_RMSE_peaks_C, templates_C, type_features


def create_global_features(meta_df, peaks_df, kmeans, kmeans_A, kmeans_B, kmeans_C, num_types_all=15, num_types_sep=6):

    X_full = process(peaks_df, meta_df)
    signal_ids = meta_df['signal_id'].values
    number_peaks_train, mean_height_peaks_train, std_height_peaks_train, \
        mean_RMSE_peaks_train, templates, number_peaks_train_A, mean_height_peaks_train_A,\
            std_height_peaks_train_A, mean_RMSE_peaks_train_A, templates_A, \
                number_peaks_train_B, mean_height_peaks_train_B, std_height_peaks_train_B,\
                    mean_RMSE_peaks_train_B, templates_B, number_peaks_train_C,\
                        mean_height_peaks_train_C, std_height_peaks_train_C, \
                            mean_RMSE_peaks_train_C, templates_C, type_features \
                                = get_features(signal_ids, kmeans, kmeans_A, kmeans_B, kmeans_C)
    # combined_train, A_train, B_train, C_train = type_features    

    for i in range(num_types_all):
        feature_name = 'number_peaks_' + str(i) 
        X_full[feature_name] = number_peaks_train[:, i]
    for i in range(num_types_sep):
        feature_name = 'number_peaks_A' + str(i) 
        X_full[feature_name] = number_peaks_train_A[:, i]
        feature_name = 'number_peaks_B' + str(i) 
        X_full[feature_name] = number_peaks_train_B[:, i]
        feature_name = 'number_peaks_C' + str(i) 
        X_full[feature_name] = number_peaks_train_C[:, i]

    for i in range(num_types_all):
        feature_name = 'mean_height_peaks_' + str(i) 
        X_full[feature_name] = mean_height_peaks_train[:, i]
    for i in range(num_types_sep):
        feature_name = 'mean_height_peaks_A' + str(i) 
        X_full[feature_name] = mean_height_peaks_train_A[:, i]
        feature_name = 'mean_height_peaks_B' + str(i) 
        X_full[feature_name] = mean_height_peaks_train_B[:, i]
        feature_name = 'mean_height_peaks_C' + str(i) 
        X_full[feature_name] = mean_height_peaks_train_C[:, i]

    for i in range(num_types_all):
        feature_name = 'std_height_peaks_' + str(i) 
        X_full[feature_name] = std_height_peaks_train[:, i]
    for i in range(num_types_sep):
        feature_name = 'std_height_peaks_A' + str(i) 
        X_full[feature_name] = std_height_peaks_train_A[:, i]
        feature_name = 'std_height_peaks_B' + str(i) 
        X_full[feature_name] = std_height_peaks_train_B[:, i]
        feature_name = 'std_height_peaks_C' + str(i) 
        X_full[feature_name] = std_height_peaks_train_C[:, i]

    for i in range(num_types_all):
        feature_name = 'mean_RMSE_peaks_' + str(i) 
        X_full[feature_name] = mean_RMSE_peaks_train[:, i]
    for i in range(num_types_sep):
        feature_name = 'mean_RMSE_peaks_A' + str(i) 
        X_full[feature_name] = mean_RMSE_peaks_train_A[:, i]
        feature_name = 'mean_RMSE_peaks_B' + str(i) 
        X_full[feature_name] = mean_RMSE_peaks_train_B[:, i]
        feature_name = 'mean_RMSE_peaks_C' + str(i) 
        X_full[feature_name] = mean_RMSE_peaks_train_C[:, i]


    feature_names = [
            'peak_count_Q13',
            #'abs_small_dist_to_min_mean_Q02',
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
            #'ratio_next_mean_Q02',
            #'ratio_prev_mean_Q02',
            'peak_count_total',
            'number_peaks_0', 'number_peaks_1', 'number_peaks_2', 'number_peaks_3', 'number_peaks_4', 'number_peaks_5', 'number_peaks_6', 'number_peaks_7', 'number_peaks_8', 'number_peaks_9', 'number_peaks_10', 'number_peaks_11', 'number_peaks_12', 'number_peaks_13', 'number_peaks_14',
            'mean_height_peaks_0', 'mean_height_peaks_1', 'mean_height_peaks_2', 'mean_height_peaks_3', 'mean_height_peaks_4', 'mean_height_peaks_5', 'mean_height_peaks_6', 'mean_height_peaks_7', 'mean_height_peaks_8', 'mean_height_peaks_9', 'mean_height_peaks_10', 'mean_height_peaks_11', 'mean_height_peaks_12', 'mean_height_peaks_13', 'mean_height_peaks_14',
           'std_height_peaks_0', 'std_height_peaks_1', 'std_height_peaks_2', 'std_height_peaks_3', 'std_height_peaks_4', 'std_height_peaks_5', 'std_height_peaks_6', 'std_height_peaks_7', 'std_height_peaks_8', 'std_height_peaks_9', 'std_height_peaks_10', 'std_height_peaks_11', 'std_height_peaks_12', 'std_height_peaks_13', 'std_height_peaks_14', 
           'mean_RMSE_peaks_0', 'mean_RMSE_peaks_1', 'mean_RMSE_peaks_2', 'mean_RMSE_peaks_3', 'mean_RMSE_peaks_4', 'mean_RMSE_peaks_5', 'mean_RMSE_peaks_6', 'mean_RMSE_peaks_7', 'mean_RMSE_peaks_8', 'mean_RMSE_peaks_9', 'mean_RMSE_peaks_10', 'mean_RMSE_peaks_11', 'mean_RMSE_peaks_12', 'mean_RMSE_peaks_13', 'mean_RMSE_peaks_14', 
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

    X_global = X_full[feature_names]

    return X_global
