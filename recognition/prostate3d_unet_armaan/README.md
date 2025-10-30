# Using a 3D UNet to segment MR images on Prostate Dataset
The goal of this project was to segment the downsampled prostate 3d MRI volumes using 3D
UNET architecture. The original dataset had 6 class labels (0-5), with 0 being the background class. The architecture used was heavily inspired by Jiangtao et al and Cicek et al, and consisted of a 4 layer encoder/decoder and a final bottleneck layer architecture with skip connections.

---
## Usage
**Dependencies**
- matplotlib==3.10.7
- nibabel==5.3.2
- numpy==2.3.4
- scipy==1.16.3
- torch==2.5.1
- torchio==0.20.23
- tqdm==4.66.5


The train and predict files have Config classes that can be adjusted by the user to set file directories. Backend architecture is handled dynamically to select CUDA or MPS based on which is available. Default file paths to Rangpur are assumed to CUDA

#### Train

```
python train.py
```

#### Test
```
python predict.py
```
## The  model
The model used was heavily inspired by Jiangtao et al. and Cicek et al. :
![Architecture](https://raw.githubusercontent.com/armaan-55/PatternAnalysis-2025/9f63a56b5a1123be8e263160653b1bea3367ca6c/recognition/prostate3d_unet_armaan/images/architecture.png)

The 3D U-Net model implemented in this project adheres to the conventional U-Net architecture, characterized by encoding and decoding paths, with skip connections between them where relevant.

At each level of encoding blocks, a 3 x 3 x 3 convolution is performed, followed by ReLU activation and BatchNorm3D. Furthermore, a 2 x 2 x 2 MaxPool3d operation is used after each level to halve the volume's spatial dimensions. The number of channels increases geometrically starting at 16 i.e. 1 --> 16 --> 32 --> 64 --> 128.

The deepest point of the network, at the bottleneck or 5th layer processes the most abstract and smallest features. It uses two consecutive 3 x 3 x 3 convolutions followed by ReLU and BatchNorm3d. Furthermore, a Dropout3d layer is included to prevent overfitting, which can be initialised / set in train.py with the dropout_p parameter (set to 0.2 for final model).

The decoder recovers the full spatial resolution and generates the final segmentation mask. It consists of four DecoderBlock levels. Each level uses 2 x 2 x 2 transposed convolution to double the spatial dimensions and upsample the data. The upsampled features are concatenated to the feature maps from the corresponding encoder. the decoder automatically pads the upsampled tensor to align with the skip connection size. Following concatenation, combined features undergo two 3 x 3 x 3 convolutions followed by ReLU and BatchNorm3d.

The final output layer maps the decoded features back to the classes using a single 1 x 1 x 1 convolution - generates 6 logits per voxel corresponding to the 6 classes, and Softmax is applied externally (in evaluation_functions.py) to obtain the final probability map, and dice loss metrics. 

### Preprocessing
#### Train-validation-test split
The dataset had 211 samples. To split the data, the best researched method was a subject level split (Rumala et al, https://www.researchgate.net/publication/374549533_How_You_Split_Matters_Data_Leakage_and_Subject_Characteristics_Studies_in_Longitudinal_Brain_MRI_Analysis). This is a common technique in medical segmentation tasks to prevent the model from memorizing patient anatomy by having multiple longitudinal values present from the same patient present in the validation or test sets.

First, the subject id's were parsed from each of the label and image files. Then, a dictionary containing the label and image for each subject was created. This dictionary of data was then split into 80 train/10 validation/10 test.

#### Standardized Preprocessing
Before training, each volume underwent fixed transformations using Torchio:
- Canonical orientation: Volumes were reoriented to a standard anatomical space.
- Resampling: Volumes were resampled to a consistent isotropic spacing.
- Intensity normalization: Image intensities were scaled to [0, 1].
- Resizing to the target shape of 192 x 192 x 96

#### Data loading and augmentation
The loading strategy differed between training and validation sets:

- The training set underwent random spatial and intensity augmentations. The spatial        augmentations were random flip, random affine, and random elastic deformation. The intensity augmentations were random bias field, random noise, and random gamma correction.

- The training loader loads full, augmented volumes. The validation and test loaders load full un-augmented volumes. This process is currently unoptimized for CUDA based machines, as training was primarily conducted locally on a MPS backend with Mac Silicone. 

## Model Evaluation
Model evaluation was done with a custom implementation in evaluation_functions.py. Here, DiceLoss and Dice Coefficient are implemented for the training.py file. For this implementation, the background class was ommitted to prevent artificial inflation of the dice score across epochs. This is because a majority of the image is background, so predicting a large portion of this correctly would artificially inflate the overall dice coefficient of the model.

The predict.py file uses a modified dice score function that includes the background class in order to get proper performance metrics across all classes.

## Training Performance

### Training/Validation Losses
The model began to converge after the 19th epoch, and scored 0.7120 on the validation set at this point. However, the model was run for a full 50 epochs to get the best possible score. This resulted in a final best model at the 38th epoch, which yielded a dice score on the validation set of 0.83, and an overall loss of 0.13. This model was then saved and used in predict.py.

The slowest classes to converge in training were classes 3 and 4, which is expected due to being finer grain structures. Interestingly, however, the lowest performing class in the prediction on unseen data was class 5. 

### Dice scores on test dataset

On the unseen test set the following results were achieved:
- Dice Coefficients for each class: \[0.994, 0.9786, 0.8856, 0.9140, 0.8286, 0.7400\]
- Overall Mean dice coefficient: 0.8902

## Segmentation Results
The following two image shows segmentation in all three dimensions:
![Result 1](https://raw.githubusercontent.com/armaan-55/PatternAnalysis-2025/9f63a56b5a1123be8e263160653b1bea3367ca6c/recognition/prostate3d_unet_armaan/images/sample_001_mpr.png)

![Result 2](https://raw.githubusercontent.com/armaan-55/PatternAnalysis-2025/9f63a56b5a1123be8e263160653b1bea3367ca6c/recognition/prostate3d_unet_armaan/images/sample_004_mpr.png)

The UNet segmentation performs uniformly well across all three axes of alignment, and the generalized performance appears to be good. Furthermore, the dice coefficients indicate incredibly strong per class segmentation. The weakest performing class was class 5, which was underrepresented in the dataset. Hence, a future improvement to the model would be to include an oversampling technique / balanced sampling technique across classes. However, as it stands, the model is well-trained.


## References
Çiçek, Ö., Abdulkadir, A., Lienkamp, Soeren S, Brox, T., & Ronneberger, O. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. ArXiv.org. https://arxiv.org/abs/1606.06650

Jiangtao et al. (2025). A Comprehensive Review of U-Net and its Variants: Advances and Applications in Medical Imaging Segmentation. ArXiv.org. https://arxiv.org/pdf/2502.06895

Rumala et al. (2023). How You Split Matters: Data Leakage and Subject Characteristics Studies in Longitudinal Brain MRI Analysis. Springer Nature. https://link.springer.com/chapter/10.1007/978-3-031-45249-9_23

COMP3710 Lecture Notes and Code Examples by Shakes Chandra