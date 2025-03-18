import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

dataset_path = "dataset"
image_size = (64, 64)

# Hyperparameter options
batch_sizes = [16]
optimizers = ['adam']
activation_functions = ['tanh']
learning_rates = [0.001]

# Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

for batch_size in batch_sizes:
    for optimizer in optimizers:
        for activation in activation_functions:
            for lr in learning_rates:
                print(f"\nTraining with Batch Size: {batch_size}, Optimizer: {optimizer}, Activation: {activation}, Learning Rate: {lr}")

                train_generator = datagen.flow_from_directory(
                    dataset_path,
                    target_size=image_size,
                    batch_size=batch_size,
                    class_mode='binary',
                    subset='training'
                )

                validation_generator = datagen.flow_from_directory(
                    dataset_path,
                    target_size=image_size,
                    batch_size=batch_size,
                    class_mode='binary',
                    subset='validation'
                )

                # CNN Model
                cnn_model = Sequential([
                    Conv2D(32, (3, 3), activation=activation, input_shape=(64, 64, 3)),
                    MaxPooling2D(2, 2),
                    Conv2D(64, (3, 3), activation=activation),
                    MaxPooling2D(2, 2),
                    Flatten(),
                    Dense(128, activation=activation),
                    Dropout(0.5),
                    Dense(1, activation='sigmoid')
                ])

                # Compile Model with custom learning rate
                opt = keras.optimizers.Adam(learning_rate=lr) if optimizer == 'adam' else \
                      keras.optimizers.SGD(learning_rate=lr) if optimizer == 'sgd' else \
                      keras.optimizers.RMSprop(learning_rate=lr)

                cnn_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

                # Train Model
                cnn_history = cnn_model.fit(
                    train_generator,
                    validation_data=validation_generator,
                    epochs=5,  # Use fewer epochs for quick testing
                    batch_size=batch_size,
                    verbose=1
                )

                # Evaluate Model
                loss, accuracy = cnn_model.evaluate(validation_generator)
                print(f"Batch: {batch_size}, Optimizer: {optimizer}, Activation: {activation}, Learning Rate: {lr} -> Accuracy: {accuracy:.2f}")

# Batch: 32, Optimizer: rmsprop, Activation: tanh, Learning Rate: 0.001 -> Accuracy: 0.97
# Batch: 16, Optimizer: adam, Activation: tanh, Learning Rate: 0.001 -> Accuracy: 0.97