

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

def load_haxby_data():
    pkl_file = "datasets.pkl"
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as f:
            training_data = pickle.load(f)
    else:
        haxby_dataset = datasets.fetch_haxby(subjects=6, fetch_stimuli=True)
        training_data = create_haxby_datasets(haxby_dataset, pkl_file, n_jobs=-1)  
    return training_data

def create_haxby_datasets(haxby_dataset, output_file, n_jobs=1):
    """
    Create sagittal/coronal/axial-slice datasets from the Haxby fMRI data and save to a pickle file.

    Parameters
    ----------
    haxby_dataset : Bunch
        The dataset object returned by nilearn.datasets.fetch_haxby.
    output_file : str
        Path to the output pickle file (e.g., 'datasets.pkl').
    n_jobs : int, default=1
        Number of parallel jobs for resizing. Use -1 to utilize all CPUs.

    The output file will contain a list of dicts, one per subject, each with keys:
      - 'X': list of (sagittal, coronal, axial) 64×64 arrays
      - 'y': list of (label, run) tuples
    """
    datasets = []
    print(f"Creating datasets for {len(haxby_dataset.func)} subjects...")

    # Loop over all subjects
    for i in range(len(haxby_dataset.func)):
        print(f"Processing subject {i + 1} of {len(haxby_dataset.func)}")
        patient = {"X": [], "y": []}

        # Load labels and run indices
        labels_df = pd.read_csv(haxby_dataset.session_target[i], sep=" ")
        y_labels = labels_df["labels"].tolist()
        runs = labels_df["chunks"].tolist()

        # Load the full 4D image once with memory mapping
        img4d = nib.load(haxby_dataset.func[i], mmap=True)
        data4d = img4d.get_fdata()

        # Precompute mid-slice indices
        x_mid, y_mid, z_mid = [d // 2 for d in img4d.shape[:3]]

        def process_volume(j):
            vol = data4d[..., j]
            sag = resize(vol[x_mid, :, :], (64, 64), preserve_range=True)
            cor = resize(vol[:, y_mid, :], (64, 64), preserve_range=True)
            axi = resize(vol[:, :, z_mid], (64, 64), preserve_range=True)
            return sag, cor, axi

        # Parallel or sequential processing
        if n_jobs == 1:
            slices = [process_volume(j) for j in range(data4d.shape[-1])]
        else:
            slices = Parallel(n_jobs=n_jobs)(
                delayed(process_volume)(j) for j in range(data4d.shape[-1])
            )

        # Collect data and labels
        patient["X"] = slices
        patient["y"] = [y_labels, runs]

        datasets.append(patient)

    # Save to disk
    with open(output_file, "wb") as f:
        pickle.dump(datasets, f)

    print(f"Saved datasets to {output_file}")
    return datasets


def create_cnn_datasets(sample_dataset):
    #Training and creating data for CNN
    # Convert the sample dataset to tensors and wrap in a DataLoader
    train_data, train_labels = sample_dataset["X"], sample_dataset["y"]
    # Extract only the string label from each tuple
    labels_str = [lbl for lbl, _ in train_labels]

    # Build a mapping from each unique label to an index
    unique_labels = sorted(set(labels_str))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    # One‐hot encode into a list of vectors
    one_hot_labels = []
    for lbl in labels_str:
        vec = [0] * len(unique_labels)
        vec[label_to_idx[lbl]] = 1
        one_hot_labels.append(vec)
        
    print("Class # to label mapping:")
    for lbl, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        print(f"Class {idx}: {lbl}")

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