# VR Mini Project 1 : Project: Face Mask Detection, Classification, and Segmentation

## Index

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Results](#results)
- [Observations and Challenges](#observations-and-challenges)
- [Steps to Run](#steps-to-run)

## Introduction

## Dataset

## Methodology

### Part A
**Dataset**: 
   - Two classes - `with_mask`, `without_mask`.  
**Preprocessing**:  
   - Convert images to **grayscale**.  
   - Resize to **64×64** pixels.  
**Feature Extraction**:  
   - Apply **Histogram of Oriented Gradients (HOG)**.  
**Models Used**:  
   - **SVM (RBF Kernel, C=1.0)**  
   - **Neural Network (MLP: 100 hidden units, ReLU, Adam optimizer, 500 epochs)**  

### Part B

### Part C

### Part D

## Experiments

### Part A
- **Train-Test Split**: 80% train, 20% test (stratified).  
- **Hyperparameter-tuning** : Hyperparameter tuning was done using grid-search cv for both the models.
- **SVM**: Trained with **RBF kernel** for non-linear separability.  
- **MLP**: One hidden layer (**100 neurons, ReLU, Adam, 500 epochs**).  

### Part B

### Part C

### Part D

## Results

### Part A
- **SVM** achieved an accuracy of **94.0%**.
- **Neural Network** achieved an accuracy of **91.0%**

### Part B

### Part C

### Part D

## Observations and Challenges

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

### Part A
- Assign the dataset path to the dataset_path variable and make sure that the subfolders present follow the naming shown above.

### Part B

### Part C

### Part D
