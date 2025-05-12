

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
from nilearn.maskers import NiftiMasker
from nilearn.input_data import NiftiMasker

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
                          smoothing_fwhm: float = 4,
                          standardize: str = 'zscore_sample',
                          n_jobs: int = 1,
                          pickle_file: bool = True,
                          output_file: str = 'haxby_slices.pkl'):
    """
    Build per-subject slice datasets from Haxby fMRI with masking, smoothing, and standardization.

    Parameters
    ----------
    haxby_dataset : Bunch
        The dataset object returned by nilearn.datasets.fetch_haxby.
    n_slices : int
        Number of equidistant slices per axis (sagittal, coronal, axial).
    output_shape : tuple
        The (height, width) to resize each 2D slice to.
    smoothing_fwhm : float
        Full-width at half-maximum for spatial smoothing (in mm).
    standardize : str
        Type of standardization ('zscore_sample' recommended).
    n_jobs : int
        Number of parallel jobs for slice extraction.
    pickle_file : bool
        Whether to pickle the resulting list of datasets.
    output_file : str
        Path for the output pickle file.

    Returns
    -------
    List[dict]
        One dict per subject with keys:
          - 'X': np.ndarray of shape (n_vols_non_rest, 3 * n_slices, H, W)
          - 'y': list of labels (length n_vols_non_rest)
    """
    datasets = []

    # Loop over each subject/session
    for func_file, targ_file in zip(haxby_dataset.func,
                                    haxby_dataset.session_target):

        # Load target labels and run indices
        df = pd.read_csv(targ_file, sep=' ')
        labels = np.array(df['labels'].tolist())
        runs = np.array(df['chunks'].tolist())

        # Remove 'rest' condition
        non_rest_mask = labels != 'rest'
        labels_no_rest = labels[non_rest_mask]

        # Initialize the masker and fit on functional 4D image
        masker = NiftiMasker(
            mask_img=haxby_dataset.mask,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            runs=runs,
            memory='nilearn_cache',
            memory_level=1
        )
        masker = masker.fit(func_file)

        # Transform to masked 2D (n_vols, n_voxels)
        masked_full = masker.transform(func_file)
        # Keep only non-rest volumes
        masked = masked_full[non_rest_mask]

        # Inverse transform back to 4D brain volumes (X, Y, Z, t_non_rest)
        masked_img = masker.inverse_transform(masked)
        data4d = masked_img.get_fdata()  # shape (X, Y, Z, n_vols_non_rest)
        n_vols = data4d.shape[-1]

        # Function to process one timepoint
        def _proc(t):
            vol3d = data4d[..., t]
            sag = split_3d_to_2d(vol3d, n_slices, output_shape, axis='x')
            cor = split_3d_to_2d(vol3d, n_slices, output_shape, axis='y')
            axi = split_3d_to_2d(vol3d, n_slices, output_shape, axis='z')
            return np.concatenate([sag, cor, axi], axis=0)

        # Parallelize slice extraction if requested
        if n_jobs == 1:
            subject_slices = [_proc(t) for t in range(n_vols)]
        else:
            subject_slices = Parallel(n_jobs=n_jobs)(
                delayed(_proc)(t) for t in range(n_vols)
            )

        X = np.stack(subject_slices, axis=0)
        y = list(labels_no_rest)

        datasets.append({'X': X, 'y': y})

    # Optionally save to pickle
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