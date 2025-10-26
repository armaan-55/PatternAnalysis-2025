import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import utils

im = utils

# Directory
base_dir = "./Prostate3D_local"

# Input MRI volumes
mr_folder = os.path.join(base_dir, "semantic_MRs")
image_names = sorted([os.path.join(mr_folder, f) for f in os.listdir(mr_folder) if f.endswith(".nii.gz")])

# Ground truth segmentation masks
label_folder = os.path.join(base_dir, "semantic_labels_only")
label_names = sorted([os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith(".nii.gz")])

def load_data_3D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    orient=False,
    early_stop=False
):
    """
    Load 3D medical image data from a list of file paths.

    Parameters:
        imageNames : list of str
            Paths to Nifti files to load.
        normImage : bool
            Normalize image to zero-mean, unit-variance.
        categorical : bool
            Convert labels to one-hot channels.
        dtype : data type
            np.float32 for images, np.uint8 for labels.
        getAffines : bool
            If True, also return affine matrices.
        orient : bool
            Apply orientation correction/resampling.
        early_stop : bool
            Stop after 20 images for quick testing/debugging.

    Returns:
        images : np.ndarray
            Array of loaded images.
        affines : list of np.ndarray (optional)
            List of affine matrices if getAffines=True.
    """

    affines = []
    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    num = len(imageNames)
    # Load first image to determine shape
    niftiImage = nib.load(imageNames[0])
    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)

    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # remove extra dimension if present

    if categorical:
        first_case = utils.to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)

        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]

        inImage = inImage[:, :, :depth].astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = utils.to_channels(inImage, dtype=dtype)
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :inImage.shape[3]] = inImage
        else:
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

def load_dataset_3D(image_names, label_names, normImage=True, dtype=np.float32, early_stop=False):
    images = load_data_3D(image_names, normImage=normImage, dtype=dtype, early_stop=early_stop)
    labels = load_data_3D(label_names, normImage=False, dtype=np.uint8, early_stop=early_stop)
    return images, labels