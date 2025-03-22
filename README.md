# VR Mini Project 1 : Project: Face Mask Detection, Classification, and Segmentation

## Index

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Results](#results)
- [Observations and Challenges](#observations-and-challenges)
- [Steps to Run](#steps-to-run)
- [Contributions](#contributions)

---

## Introduction

---

## Dataset

---

## Methodology

 #### Part A
 The following is done:
**Preprocessing**:  
   - Convert images to grayscale.  
   - Resize to 64×64 pixels.  
**Feature Extraction**:  
   - Apply istogram of Oriented Gradients (HOG).  
**Models Used**:  
   - SVM (RBF Kernel, C=1.0)  
   - Neural Network (MLP: 100 hidden units, ReLU, Adam optimizer, 500 epochs) 

#### Part B
The following is done:
**Preprocessing**:  
   - Rescaled images (`1/255`) for normalization.  
   - 80-20 split for training & validation.  
**CNN Architecture**:  
   - Conv2D layers with Tanh activation.  
   - Pooling & Dropout for regularization.  
   - Fully connected layers & Sigmoid activation for classification.  
**Hyperparameters**:  
   - Batch Size: 16
   - Optimizer: Adam
   - Activation: Tanh 
   - Learning Rate: 0.001

#### Part C
Here we tried the following methods:
- **kmeans**: Run a simple kmeans clustering based on each pixel's rgb values with k=2
- **gmm**: Again a simple clustering algorithm that is run with n_components = 2
- **thresholding**: Here we use otsu's method of thresholding. 
- **watershed**: Use the watershed algorithm and morphological operations to segment the image. More than just mask and non-mask regions may form.
- **canny**: Use the canny edge detector to detect edges. Post this we consider two methods.
The first one involves detecting horizontal and vertical coordinates where edges are prominent. The regions between the prominent 
edges is then filled to obtain the segmented output.
The next one involves running a bfs and considering large edges only. Then using these edges,
minimum and maximum x and y coordinates are chosen to approximate the location of the mask.
If edges are prominent in other parts of the image such other than at the mask face boundary of within the mask, we may go wrong.

All outputs can be visualized using the code by setting the paramater show to be equal to True in the segmentation function.

#### Part D

---

## Experiments

#### Part A
- **Train-Test Split**: 80% train, 20% test (stratified).  
- **Hyperparameter-tuning** : Hyperparameter tuning was done using grid-search cv for both the models.
- **SVM**: Trained with RBF kernel for non-linear separability.  
- **MLP**: One hidden layer (100 neurons, ReLU, Adam, 500 epochs).  

#### Part B
- Trained the CNN model using different optimizers, batch sizes, and activation functions.
- Evaluated the accuracy for 5 epochs on the validation set.

#### Part C
- kmeans, gmm and otsu thresholding were straight forward to code and required no specific experiments.
- Using the watershed algorithm involved tweaking the morphological operation (CLOSE OR OPEN), the threshold limits 
   and the size of the blurring kernel. However no significant improvement was observed. 
   If it performed better on one image, it didn't perform as well on another.
   Watershed can't be scored using something like IOU as it is possible that more than two regions are obtained.
   Assigning these as mask and not mask needs manual intervention.
- Canny is an edge detector and not for segmentation. So, coming with techniques to use this method to aid in 
   segmentation was not a trivial task. Although the methods used here are not robust algorithms with formal proofs,
   they perform at a level comparable to the other methods.


#### Part D

---

## Results

#### Part A
- SVM achieved an accuracy of 94.0%.
- Neural Network achieved an accuracy of 91.0%

#### Part B
- The CNN model gave the highest accuracy of 97% compared to the models present in the part A.
- SVM and MLP may work well with smaller datasets. They donot capture the spatial feaures as CNN does which helps learning edge, texture, and
  shaper hierarchies through the convolution layers.
- SVM and MLP require manual feature extraction which may not capture complex features efficiently.

#### Part C
All the methods mentioned are subject to error. This is due to the images having different lighting conditions, contrasts,
colors, gradients, designs on masks etc. Thresholding inevitably fails for images taken in different conditions.
Sometimes the surrouding image has a similar color to that of the mask. This makes kmeans and gmm produce severely inaccurate results.
The method used to segment using canny edges is purely a heuristic and has poor performance especially when areas other than the mask have edges.
The scores are printed when the code is run. IOU is used the metric. Alternatively, dice scores could also be used.
The highest IOU score achieved was around 0.86 by gmm. IOU scores as low as 0.2 were encountered on blurred images.

#### Part D

---

## Observations and Challenges
CNN model shows to outperform the other models such as the SVM or MLP due to its ability to capture the spatial features and well generalization.
Supervised machine learning methods perform much better than traditional methods in the segmentation task.

---

## Steps to Run

Add the datasets so that the directory structure looks as follows:

```
📂a
📂b
📂c
📂d
📂datasets
 ┣ 📂dataset1
 ┃ ┣ 📂without_mask
 ┃ ┗ 📂with_mask
 ┗ 📂dataset2
 ┃ ┣ 📂1
 ┃ ┃ ┣ 📂face_crop
 ┃ ┃ ┣ 📂face_crop_segmentation
 ┃ ┣ 📂2
 ┃ ┃ ┣ 📂img
```

Ensure you have `python 3.10`. To install required libraries, run 

```
pip install requirements.txt
```

#### Part A
- Assign the dataset path to the dataset_path variable and make sure that the subfolders present follow the naming shown above.

#### Part B

#### Part C
- Change the path_to_data and path_to_segmented_op variables to represent the directories in which the input images and segmentation ground truth masks are present.
- The values are already filled correctly if the directory structure mentioned above is followed.

#### Part D

---

## Contributions

---

Github Link : 
