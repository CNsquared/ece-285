

import nibabel as nib
from skimage.transform import resize
from joblib import Parallel, delayed
import pandas as pd
import pickle
from nilearn import datasets
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_haxby_data(n_slices=3, pickle_file = True):
    pkl_file = "datasets.pkl"
    if os.path.exists(pkl_file):
        
        with open(pkl_file, "rb") as f:
            training_data = pickle.load(f)
    else:
        haxby_dataset = datasets.fetch_haxby(subjects=6, fetch_stimuli=True)
        training_data = create_haxby_datasets(haxby_dataset, output_file=pkl_file, n_jobs=-1, n_slices=n_slices, pickle_file=pickle_file)  
    return training_data
import os
import pickle

import numpy as np
import pandas as pd
import nibabel as nib
from skimage.transform import resize
from joblib import Parallel, delayed


def split_3d_to_2d(volume_3d: np.ndarray,
                   n_slices: int = 3,
                   output_shape: tuple = (64, 64),
                   axis: str = 'z') -> np.ndarray:
    """
    Extract `n_slices` equidistant 2D slices from a 3D volume along `axis`,
    resize each to `output_shape`, and return an array of shape (n_slices, H, W).
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax = axis_map[axis.lower()]
    dim = volume_3d.shape[ax]
    idxs = np.linspace(0, dim - 1, n_slices, dtype=int)

    slices = []
    for i in idxs:
        if ax == 0:
            sl = volume_3d[i, :, :]
        elif ax == 1:
            sl = volume_3d[:, i, :]
        else:
            sl = volume_3d[:, :, i]
        sl = resize(
            sl,
            output_shape,
            preserve_range=True,
            anti_aliasing=True
        ).astype(volume_3d.dtype)
        slices.append(sl)
    return np.stack(slices, axis=0)


def create_haxby_datasets(haxby_dataset,
                          n_slices: int = 3,
                          output_shape: tuple = (64, 64),
                          n_jobs: int = 1,
                          pickle_file: bool = True,
                          output_file: str = 'haxby_slices.pkl'):
    """
    Build per-subject slice datasets from Haxby fMRI.

    Returns
    -------
    List[dict], one per subject, each with:
      - 'X': np.ndarray of shape (n_vols, 3 * n_slices, 64, 64)
      - 'y': list of length n_vols with the label for each volume
    """
    datasets = []

    for func_file, targ_file in zip(haxby_dataset.func,
                                    haxby_dataset.session_target):

        # Load labels
        df = pd.read_csv(targ_file, sep=' ')
        labels = df['labels'].tolist()
        


        # Load 4D fMRI
        img4d = nib.load(func_file)
        data4d = img4d.get_fdata()        # shape (X, Y, Z, T)
        n_vols = data4d.shape[-1]

        # Process one volume: get n_slices per axis, then concatenate
        def _proc(t):
            vol3d = data4d[..., t]
            sag = split_3d_to_2d(vol3d, n_slices, output_shape, axis='x')
            cor = split_3d_to_2d(vol3d, n_slices, output_shape, axis='y')
            axi = split_3d_to_2d(vol3d, n_slices, output_shape, axis='z')
            # concatenate into (3*n_slices, H, W)
            return np.concatenate([sag, cor, axi], axis=0)

        # Run processing
        if n_jobs == 1:
            subject_slices = [_proc(t) for t in range(n_vols)]
        else:
            subject_slices = Parallel(n_jobs=n_jobs)(
                delayed(_proc)(t) for t in range(n_vols)
            )

        # Stack to (n_vols, 3*n_slices, H, W)
        X = np.stack(subject_slices, axis=0)
        y = labels

        datasets.append({'X': X, 'y': y})

    # Optionally pickle
    if pickle_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(datasets, f)
        print(f"Saved {len(datasets)} subject datasets to {output_file}")

    return datasets


def create_cnn_datasets(sample_dataset):
    #Training and creating data for CNN
    # Convert the sample dataset to tensors and wrap in a DataLoader
    train_data, train_labels = sample_dataset["X"], sample_dataset["y"]
    # Extract only the string label from each tuple
    labels_str = [lbl for lbl in train_labels]

    # Build a mapping from each unique label to an index
    unique_labels = sorted(set(labels_str))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    # One‐hot encode into a list of vectors
    one_hot_labels = []
    for lbl in labels_str:
        vec = [0] * len(unique_labels)
        vec[label_to_idx[lbl]] = 1
        one_hot_labels.append(vec)
        
    #print("Class # to label mapping:")
    #for lbl, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
    #    print(f"Class {idx}: {lbl}")

    # Replace train_labels with the one‐hot encoded vectors
    train_labels = one_hot_labels
    
    # Convert list of arrays to numpy arrays for better performance
    # Convert to numpy arrays
    data_arr = np.array(train_data)
    labels_arr = np.array(train_labels)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data_arr, labels_arr, test_size=0.2, shuffle=True, random_state=42
    )

    # Convert to tensors
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_labels_tensor = torch.tensor(y_train, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_labels_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets
    train_dataset = TensorDataset(train_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_tensor, test_labels_tensor)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    
    return train_loader, test_loader

if __name__ == "__main__":
    haxby_dataset = datasets.fetch_haxby(subjects=6, fetch_stimuli=True)
    training_data = create_haxby_datasets(haxby_dataset, "datasets.pkl", n_jobs=-1)