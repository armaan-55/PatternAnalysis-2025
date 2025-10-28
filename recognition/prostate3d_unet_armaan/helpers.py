import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
import torchio as tio
import torch
import random

def to_channels(label_volume, num_classes=None, dtype=np.uint8):
    if num_classes is None:
        num_classes = int(label_volume.max() + 1)
    shape = label_volume.shape + (num_classes,)
    out = np.zeros(shape, dtype=dtype)
    for c in range(num_classes):
        out[..., c] = (label_volume == c).astype(dtype)
    return out

def applyOrientation(nifti_img, interpolation='linear', scale=1):
    """Reorient a NiFTI image to the closest canonical orientation."""
    reoriented_img = nib.as_closest_canonical(nifti_img)
    return reoriented_img

def resize_image(image_data, original_affine, target_shape, interpolation_order=1):
    """Resizes a 3D image to the target shape using zoom interpolation."""
    current_shape = image_data.shape
    if current_shape == target_shape:
        return image_data
    scale_factors = [n / o for n, o in zip(target_shape, current_shape)]
    return zoom(image_data, scale_factors, order=interpolation_order)

class ImageProcessor:
    """
    Handles all pre-processing and on-the-fly augmentation for a single 
    image/label pair using Torchio.
    """
    def __init__(self, target_shape=(128, 128, 128)):
        self.target_shape = target_shape
        self.preprocessing = tio.Compose([
            tio.ToCanonical(),
            tio.Resize(target_shape),
        ])

        # Random transformations for use on training data
        self.augment_transforms = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
            
            # Affine/Rotation (Geometric distortion)
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=10,
                translation=5,
                isotropic=True,
                p=0.5,
            ),
            
            # Elastic Deformation (Non-linear deformation)
            tio.RandomElasticDeformation(
                num_control_points=5,
                max_displacement=7.5,
                p=0.25,
            )
        ])
    
    @staticmethod
    def load_nifti(image_path, dtype=np.float32):
        """Load Nifti file and return data array and affine."""
        nifti_image = nib.load(image_path)
        image_data = nifti_image.get_fdata().astype(dtype)
        if len(image_data.shape) == 4:
            image_data = image_data[..., 0]
        return image_data, nifti_image.affine

    def process_pair(self, mri_path, label_path, is_augmenting=True):
        """
        Loads, preprocesses, and augments a single image/label pair.
        Returns a (C, D, H, W) image tensor and a (D, H, W) label tensor.
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

        # Apply FIXED Preprocessing (Orientation, Resizing)
        subject = self.preprocessing(subject)
        
        # Apply RANDOM Augmentation
        if is_augmenting:
            # Torchio applies spatial transforms (flip, rotation) to both
            subject = self.augment_transforms(subject)
            # Intensity transforms are only applied to the image (ScalarImage)
            subject['image'] = self.random_intensity_transforms()(subject['image'])

        # Final Normalization and Tensor Conversion
        mri_data_out = subject['image'].data.squeeze().numpy()
        label_data_out = subject['label'].data.squeeze().numpy()
        
        # Z-score normalization
        mri_data_out = (mri_data_out - np.mean(mri_data_out)) / (np.std(mri_data_out) + 1e-8)
        
        # Final PyTorch Tensor format: (C, D, H, W) for image, (D, H, W) for label
        mri_tensor_out = torch.tensor(mri_data_out, dtype=torch.float32).unsqueeze(0)
        label_tensor_out = torch.tensor(label_data_out, dtype=torch.long) 
        
        return mri_tensor_out, label_tensor_out