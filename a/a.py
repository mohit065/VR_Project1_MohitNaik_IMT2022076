import os
import cv2
import numpy as np
from skimage.feature import hog
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Function to extract HOG features
def extract_hog_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (64, 64))  # Resize for consistency
    features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                    orientations=9, block_norm='L2-Hys', visualize=False)
    return features

# Function to load dataset in parallel
def process_image(file_path, label):
    image = cv2.imread(file_path)
    if image is None:
        return None, None
    features = extract_hog_features(image)
    return features, label if features is not None else (None, None)

def load_dataset(dataset_path):
    X, y = [], []
    categories = ['with_mask', 'without_mask']
    all_files = []
    for label, category in enumerate(categories):
        folder_path = os.path.join(dataset_path, category)
        if not os.path.exists(folder_path):
            continue
        for file in os.listdir(folder_path):
            all_files.append((os.path.join(folder_path, file), label))

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda p: process_image(*p), all_files))

    X, y = zip(*[(f, l) for f, l in results if f is not None])
    return np.array(X), np.array(y)

# Main execution
if __name__ == "__main__":
    dataset_path = os.path.abspath("../datasets/dataset1")  # Ensure absolute path
    X, y = load_dataset(dataset_path)

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Optimized SVM model (increased cache size, better gamma)
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', cache_size=700)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)

    # Optimized Neural Network model (faster solver, increased max_iter)
    nn_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300)
    nn_model.fit(X_train, y_train)
    y_pred_nn = nn_model.predict(X_test)
    nn_accuracy = accuracy_score(y_test, y_pred_nn)

    # Print the accuracies
    print(f"SVM Accuracy: {svm_accuracy:.2f}")
    print(f"Neural Network Accuracy: {nn_accuracy:.2f}")