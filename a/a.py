import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# This function extracts the HOG features from the input image
def extract_hog_features(image):
    # Preprocessing of the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converting the image to grayscale
    image = cv2.resize(image, (64, 64))  # Resize for uniformity
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                      orientations=9, block_norm='L2-Hys', visualize=True)
    return features

# This function loads the dataset as per the format given in the readme
def load_dataset(dataset_path):
    X, y = [], []
    categories = ['with_mask', 'without_mask']
    
    for label, category in enumerate(categories):
        folder_path = os.path.join(dataset_path, category)
        for file in tqdm(os.listdir(folder_path), desc=f"Processing {category}"):
            img_path = os.path.join(folder_path, file)
            image = cv2.imread(img_path)
            if image is not None:
                features = extract_hog_features(image)
                X.append(features)
                y.append(label)
    
    return np.array(X), np.array(y)

# Load dataset path 
dataset_path = "dataset"
X, y = load_dataset(dataset_path)

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameters used for tuning the model
# param_grid = {
#     'C': [0.1, 1, 10, 100],  # Regularization strength
#     'kernel': ['linear', 'rbf', 'poly'],
#     'gamma': ['scale', 'auto']
# }

# Train and evaluate the SVM model
svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

# Hyperparameters used for tuning the model
# param_grid = {
#     'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],  # Different layer configurations
#     'activation': ['relu', 'tanh', 'logistic'],  # Different activation functions
#     'solver': ['adam', 'sgd', 'lbfgs'],  # Different solvers
#     'alpha': [0.0001, 0.001, 0.01],  # L2 Regularization
#     'learning_rate': ['constant', 'adaptive'],  # Learning rate strategy
#     'max_iter': [500, 1000]  # Iteration limits
# }

# Train and evaluate the Neural Network model
nn_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)
nn_accuracy = accuracy_score(y_test, y_pred_nn)

# Pint the accuracy of both the models
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"Neural Network Accuracy: {nn_accuracy:.2f}")