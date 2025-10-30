import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
import torchio as tio
import torch
import random

"""
helpers.py - Utility / helper nethods for other files.

Contains methods for image transformation on the fly for improved training handled using
the ImageProcessor class.

Author: Armaan Aulakh
Date: October 30 2025
"""

def to_channels(label_volume, num_classes=None, dtype=np.uint8):
    """
    Converts a sparse index map volume into a one-hot encoded channel format.

    Args:
        label_volume (np.ndarray): The input label map with shape (D, H, W).
        num_classes (int, optional): The total number of classes (C). If None, 
                                     it is inferred from `label_volume.max() + 1`.
        dtype (type, optional): Data type for the output array. Defaults to np.uint8.

    Returns:
        np.ndarray: The one-hot encoded volume with shape (D, H, W, C).
    """
    if num_classes is None:
        num_classes = int(label_volume.max() + 1)
    shape = label_volume.shape + (num_classes,)
    out = np.zeros(shape, dtype=dtype)
    for c in range(num_classes):
        out[..., c] = (label_volume == c).astype(dtype)
    return out

def applyOrientation(nifti_img, interpolation='linear', scale=1):
    """
    Reorient a NiFTI image to the closest canonical orientation using nibabel.as_closest_canonical
    Ensures spatial orientation is consistent across input volumes.

    Args:
        nifti_img(nibabel.Nifti1image): Input image object.
        interpolation (str, optional): Interpolation method.
        scale (int, optional): Scaling factor.

    Returns:
        nibabel.Nifti1Image: Reoriented image object.
    """
    reoriented_img = nib.as_closest_canonical(nifti_img)
    return reoriented_img

def resize_image(image_data, original_affine, target_shape, interpolation_order=1):
    """
    Resizes a 3D image to the target shape using `scipy.ndimage.zoom` interpolation.

    Args:
        image_data (np.ndarray): The input image array.
        original_affine (np.ndarray): The affine matrix.
        target_shape (tuple): The desired output shape (D, H, W).
        interpolation_order (int, optional): The order of spline interpolation (0=nearest, 1=linear). 
                                             Defaults to 1 (linear).

    Returns:
        np.ndarray: The resized image array.
    """
    current_shape = image_data.shape
    if current_shape == target_shape:
        return image_data
    scale_factors = [n / o for n, o in zip(target_shape, current_shape)]
    return zoom(image_data, scale_factors, order=interpolation_order)

class ImageProcessor:
    """
    Manages the complete data pipeline for a single image/label pair.

    Utilizes Torchio to handle standardized pre-processing (reorientation, 
    resizing, intensity normalization) and on-the-fly random spatial and 
    intensity augmentations for training.
    """
    def __init__(self, target_shape=(128, 128, 128), num_classes=6):
        """
        Initializes the processor with target dimensions and defines the Torchio 
        transformation pipelines.

        Args:
            target_shape (tuple, optional): The final desired spatial size (D, H, W). 
                                            Defaults to (128, 128, 128).
            num_classes (int, optional): The total number of classes. Defaults to 6.
        """
        self.target_shape = target_shape
        self.num_classes = num_classes
        self.preprocessing = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(1.0),
            tio.Resize(target_shape),
            tio.RescaleIntensity(out_min_max=(0,1)),
        ])

        # Random spatial transformations for training data
        self.augment_transforms = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=10,
                translation=5,
                isotropic=True,
                p=0.5,
            ),
            tio.RandomElasticDeformation(
                num_control_points=5,
                max_displacement=7.5,
                p=0.25,
            )
        ])

    # Separate function for random intensity transforms (applied only to image)
    def random_intensity_transforms(self):
        """
        Defines a Torchio pipeline for random intensity augmentations.

        Includes: Random Bias Field, Random Noise, and Random Gamma correction.

        Returns:
            tio.Compose: The Torchio transformation object.
        """
        return tio.Compose([
            tio.RandomBiasField(p=0.2),
            tio.RandomNoise(p=0.2),
            tio.RandomGamma(p=0.2),
        ])

    @staticmethod
    def load_nifti(image_path, dtype=np.float32):
        """
        Loads a NIfTI file, extracts the image data, and handles 4D volumes.

        Args:
            image_path (str): Full path to the NIfTI file (.nii or .nii.gz).
            dtype (type, optional): Data type for the output array. Defaults to np.float32.

        Returns:
            tuple: (image_data, nifti_affine)
                - image_data (np.ndarray): The 3D image array.
                - nifti_affine (np.ndarray): The affine matrix defining spatial orientation.
        """        
        nifti_image = nib.load(image_path)
        image_data = nifti_image.get_fdata().astype(dtype)

        # Handle 4D data (e.g., if a single time point is present)
        if len(image_data.shape) == 4:
            image_data = image_data[..., 0]
        return image_data, nifti_image.affine

    def process_pair(self, mri_path, label_path, is_augmenting=True):
        """
        Loads, preprocesses, and conditionally augments a single image/label pair.

        Args:
            mri_path (str): Path to the input MRI volume.
            label_path (str): Path to the corresponding label map.
            is_augmenting (bool, optional): If True, applies random spatial and 
                                           intensity transforms. Defaults to True.

        Returns:
            tuple: (mri_tensor_out, label_tensor_out)
                - mri_tensor_out (torch.Tensor): The preprocessed image tensor (1, D, H, W).
                - label_tensor_out (torch.Tensor): The preprocessed label tensor (D, H, W) of type long.
        """
        # Load data
        mri_data, mri_affine = self.load_nifti(mri_path, dtype=np.float32)
        label_data, label_affine = self.load_nifti(label_path, dtype=np.uint8)

        # Create Torchio Subject
        mri_tensor = torch.tensor(mri_data).unsqueeze(0)
        label_tensor = torch.tensor(label_data).unsqueeze(0)

        # Use appropriate tio classes for interpolation
        mri_subject = tio.ScalarImage(tensor=mri_tensor, affine=mri_affine)
        label_subject = tio.LabelMap(tensor=label_tensor, affine=label_affine)
        subject = tio.Subject(image=mri_subject, label=label_subject)

        # Apply fixed Preprocessing (Orientation, Resizing)
        subject = self.preprocessing(subject)

        # Apply random Augmentation
        if is_augmenting:
            # Torchio applies spatial transforms to both
            subject = self.augment_transforms(subject)
            # Intensity transforms are only applied to the image (ScalarImage)
            subject['image'] = self.random_intensity_transforms()(subject['image'])

        # Final Normalization and Tensor Conversion
        mri_data_out = subject['image'].data.squeeze().numpy()
        label_data_out = subject['label'].data.squeeze().numpy()

        # Final PyTorch Tensor format.
        mri_tensor_out = torch.tensor(mri_data_out, dtype=torch.float32).unsqueeze(0)
        label_tensor_out = torch.tensor(label_data_out, dtype=torch.long)

        unique_labels = torch.unique(label_tensor_out)
        if torch.any(unique_labels >= self.num_classes) or torch.any(unique_labels < 0):
            print(f"WARNING: Invalid label values found: {unique_labels}")
    
        return mri_tensor_out, label_tensor_out