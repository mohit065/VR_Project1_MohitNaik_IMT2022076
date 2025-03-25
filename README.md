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

The aim of this project was to perform facemask detection, classification and segmentation given a couple of facemask datasets. Specifically, the objectives were:
- Binary classification of images (with or without mask) using handcrafted features and traditional models, as well as using CNNs.
- Facemask segmentation and evaluation using traditional methods like GMMs, K-Means, Watershed etc. as well as using U-Net.

---

## Dataset

- The first dataset, used for classification tasks, contains around 2000 images each of faces with and without masks. All images have variations in size, illumination, gender, pose, etc.
- The second dataset, used for segmentation tasks, contains around 9000 images, again in various sizes and lighting conditions, of people wearing a face mask. For each image, the ground truth segmented output with the facemmask region coloured white, and the remaining region black is also provided.

---

## Methodology

### Classification Tasks

For the handcrafted features part, we did the following:

**Preprocessing**: Converted images to grayscale and resized to 64×64 pixels.

**Feature Extraction**: Applied Histogram of Oriented Gradients (HOG) to capture edge and texture information by computing gradient orientations in localized regions of the image, emphasizing shape and structure for effective classification.

**Models Used**: We used an SVM classifier with an RBF kernel, along with a Multilayer Perceptron with 100 hidden layers, ReLU activation, optimized using Adam.

For the CNN part, we did the following:

**CNN Architecture**: 2 convolutional layers with 32 and 64 filters respectively along with max pooling, followed by a dense layer with 128 filters and then a dropout layer. Other hyperparameters like learning rates, activations, epochs and batch sizes were tuned and selected.

### Segmentation

We tried the following traditional methods:

- K-Means: Performed K-Means clustering on pixel RGB values to segment the image into two clusters, aiming to distinguish mask and non-mask regions.
- GMM: Applied Gaussian Mixture Models for clustering, treating the image as a mixture of two distributions to separate the facemask region.
- Thresholding: Used Otsu’s method to determine an optimal threshold value for segmenting the facemask region from the background.
- Watershed: Utilized the Watershed algorithm combined with morphological operations to segment the image. This method may produce more than just mask and non-mask regions.
- Canny: Apply the Canny edge detector to identify edges, followed by two different approaches for segmentation:
   - Detect prominent horizontal and vertical edges, then fill regions between them to obtain the segmented mask.
   - Perform a breadth-first search (BFS) to identify large edges, then use the minimum and maximum x and y coordinates of these edges to approximate the mask’s location. If strong edges appear outside the mask region, segmentation accuracy may be affected.

---

For the U-Net part, we did the following:

**Preprocessing**: Converted images to grayscale and resized to 64×64 pixels.
**Architecture**: The model follows a standard U-Net architecture with an encoder and a decoder part, and skip connections for preserving spatial features.
 - The encoder block (downsampling) has 4 separable convolutional blocks with 32, 64, 128 and 512 filters respectively, with max pooling after each of the first 3 layers. The final convolution layer is the bottleneck layer, bridging the encoder and decoder while capturing high-level features
 - The decoder block upsamples the feature maps back to the size of the input using 3 transposed convolution layers with 128, 64 and 32 filters respectively, and restores details using skip connections. A final convolution layer with sigmoid activation + thresholding is used to produce a binary mask that segments the facemask region.

## Experiments

### Classification

All 3 models were thoroughly evaluated with a variety of hyperparameters like number of layers, activations, kernels, number of epochs, optimizers, batch sizes, learning rates, number of filters etc. 

### Segmentation

- kmeans, gmm and otsu thresholding were straight forward to code and required no specific experiments.
- Using the watershed algorithm involved tweaking the morphological operation (CLOSE OR OPEN), the threshold limits and the size of the blurring kernel. However no significant improvement was observed. If it performed better on one image, it didn't perform as well on another.
- Watershed can't be scored using something like IOU as it is possible that more than two regions are obtained. Assigning these as mask and not mask needs manual intervention.
- Canny is an edge detector and not for segmentation. So, coming with techniques to use this method to aid in segmentation was not a trivial task. Although the methods used here are not robust algorithms with formal proofs, they perform at a level comparable to the other methods.

---

## Results

### Classification

- SVM achieved an accuracy of 94.0%.
- Neural Network achieved an accuracy of 91.0%
- The CNN model gave the highest accuracy of 97%.
- SVM and MLP may work well with smaller datasets. They donot capture the spatial feaures as CNN does which helps learning edge, texture, and shape hierarchies through the convolution layers.
- SVM and MLP require manual feature extraction which may not capture complex features efficiently.

### Segmentation

All the methods mentioned are subject to error. This is due to the images having different lighting conditions, contrasts,
colors, gradients, designs on masks etc. Thresholding inevitably fails for images taken in different conditions.
Sometimes the surrouding image has a similar color to that of the mask. This makes kmeans and gmm produce severely inaccurate results.
The method used to segment using canny edges is purely a heuristic and has poor performance especially when areas other than the mask have edges.
The scores are printed when the code is run. IOU is used the metric. Alternatively, dice scores could also be used.
The highest IOU score achieved was around 0.86 by gmm. IOU scores as low as 0.2 were encountered on blurred images.

---

## Observations and Challenges

CNN model shows to outperform the other models such as the SVM or MLP due to its ability to capture the spatial features and well generalization.
Supervised machine learning methods perform much better than traditional methods in the segmentation task.

---

## Steps to Run

Add the datasets so that the directory structure looks as follows:

```none
📂classification
📂segmentation
📂datasets
 ┣ 📂dataset1
 ┃ ┣ 📂without_mask
 ┃ ┗ 📂with_mask
 ┣ 📂dataset2
 ┃ ┣ 📂face_crop
 ┗ ┗ 📂face_crop_segmentation
```

Ensure you have `python 3.10`. To install required libraries, run

```none
pip install -r requirements.txt
```

after which you can run the notebooks.

---

## Contributions

- IMT2022017 Prateek Rath : Wrote the code for Part C (Segmentation using traditional methods)
- IMT2022076 Mohit Naik : Wrote the code for Part D (Segmentation using U-Net)
- IMT2022519 Vedant Mangrulkar : Wrote the code for Parts A and B (Classification Tasks)

---

Github Link : https://github.com/mohit065/VR_Project1_MohitNaik_IMT2022076
