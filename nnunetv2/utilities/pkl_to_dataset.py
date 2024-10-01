import pickle
import numpy as np
import argparse
import glob
import os
import json

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results

def from_one_hot_to_indices(segmentation):
    indices = np.zeros(segmentation.shape[1:])
    indices[segmentation[0] == 1] = 1
    indices[segmentation[1] == 1] = 2
    indices[segmentation[2] == 1] = 3
    indices[segmentation[4] == 1] = 4
    return indices

def pkl_to_dataset(pkl_paths, folder=""):
    if not isinstance(pkl_paths, list):
        pkl_paths = [pkl_paths]

    data = []
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            data += pickle.load(f)

    if folder == "":
        datasets = glob.glob(nnUNet_raw+'/Dataset*')
        ids = [int(d.split('/')[-1].split('Dataset')[1].split('_')[0]) for d in datasets]
        ids.sort()
        dataset_id = ids[-1] + 1
        base = "Dataset%03.0d" % dataset_id
        for pkl_path in pkl_paths:
            base += "_"+pkl_path.split('/')[-1].split('.')[0]
        folder = base

    basedir = 'nnUNet_raw/'
    if folder != "" and not os.path.exists(os.path.join(basedir, folder)):
        os.makedirs(os.path.join(basedir, folder))

    if not os.path.exists(os.path.join(basedir, folder, 'imagesTr')):
        os.makedirs(os.path.join(basedir, folder, 'imagesTr'))
    if not os.path.exists(os.path.join(basedir, folder, 'labelsTr')):
        os.makedirs(os.path.join(basedir, folder, 'labelsTr'))

    nfiles = 0
    for record in data:
        signal = record["signal"]
        segmentation = from_one_hot_to_indices(record["segmentation"])
        haslabel = record["is_labeled"]
        db = record["db"]

        if haslabel and db != "STANFORD":
            nfiles += 1
            np.save(os.path.join(basedir, folder, 'imagesTr', f'case_{record["record"]}_0000.npy'), signal)
            np.save(os.path.join(basedir, folder, 'labelsTr', f'case_{record["record"]}.npy'), segmentation)

    jsn = {
            "channel_names": {
                "0": "LeadII"
            },
            "labels": {
                "background": 0,
                "p_wave": 1,
                "qrs_wave": 2,
                "t_wave": 3,
                "noise": 4
            },
            "numTraining": nfiles,
            "file_ending": ".npy"
        }

    with open(os.path.join(basedir, folder, 'dataset.json'), 'w') as f:
        json.dump(jsn, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a pkl file to a dataset')
    parser.add_argument('-ps','--pkl_paths', nargs='+', help='Path(s) to the pkl file', required=True)
    args = parser.parse_args()
    pkl_to_dataset(args.pkl_paths)