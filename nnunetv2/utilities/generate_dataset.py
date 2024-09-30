import numpy as np
import os
import neurokit2 as nk
import matplotlib.pyplot as plt

def simulate_ecg(length, fs):
    # Simulate ECG signal
    hr = np.random.randint(40, 100)
    ecg = nk.ecg_simulate(duration=length, heart_rate=hr, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=fs)
    _, waves = nk.ecg_delineate(ecg, rpeaks, sampling_rate=fs)

    p_onsets = waves['ECG_P_Onsets']
    p_offsets = waves['ECG_P_Offsets']
    t_onsets = waves['ECG_T_Onsets']
    t_offsets = waves['ECG_T_Offsets']
    qrs_onsets = waves['ECG_R_Onsets']
    qrs_offsets = waves['ECG_R_Offsets']

    # Simulate segmentation
    seg = np.zeros(length*fs)
    for i in range(len(p_onsets)):
        if not np.isnan(p_onsets[i]) and not np.isnan(p_offsets[i]):
            seg[p_onsets[i]:p_offsets[i]] = 1
    for i in range(len(qrs_onsets)):
        if not np.isnan(qrs_onsets[i]) and not np.isnan(qrs_offsets[i]):
            seg[qrs_onsets[i]:qrs_offsets[i]] = 2
    for i in range(len(t_onsets)):
        if not np.isnan(t_onsets[i]) and not np.isnan(t_offsets[i]):
            seg[t_onsets[i]:t_offsets[i]] = 3

    return ecg, seg

def generate_dataset(n_files=10, length=2048, fs=200, folder=""):
    dataset = {}
    basedir = '/Users/lukasarts/Dropbox/UU/ASRA/nnUNet/nnUNet_raw/'
    if folder != "" and not os.path.exists(os.path.join(basedir, folder)):
        os.makedirs(os.path.join(basedir, folder))

    if not os.path.exists(os.path.join(basedir, folder, 'imagesTr')):
        os.makedirs(os.path.join(basedir, folder, 'imagesTr'))
    if not os.path.exists(os.path.join(basedir, folder, 'labelsTr')):
        os.makedirs(os.path.join(basedir, folder, 'labelsTr'))

    for i in range(n_files):
        sig, seg = simulate_ecg(length, fs)
        dataset[f'case_{i}'] = {'data': sig, 'seg': seg}
        np.save(os.path.join(basedir, folder, 'imagesTr', f'case_{i}_0000.npy'), sig)
        np.save(os.path.join(basedir, folder, 'labelsTr', f'case_{i}.npy'), seg)

if __name__ == '__main__':
    n_files = 25
    length = 10
    fs = 200
    folder = 'Dataset0011_test'
    generate_dataset(n_files, length, fs, folder=folder)




