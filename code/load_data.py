

import nibabel as nib
from skimage.transform import resize
from joblib import Parallel, delayed
import pandas as pd
import pickle
from nilearn.datasets import fetch_haxby
from nilearn import datasets
from nilearn.plotting import show

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
        patient["y"] = list(zip(y_labels, runs))

        datasets.append(patient)

    # Save to disk
    with open(output_file, "wb") as f:
        pickle.dump(datasets, f)

    print(f"Saved datasets to {output_file}")
    return datasets

if __name__ == "__main__":
    haxby_dataset = datasets.fetch_haxby(subjects=6, fetch_stimuli=True)
    training_data = create_haxby_datasets(haxby_dataset, "datasets.pkl", n_jobs=-1)