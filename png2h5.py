#
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array

# Set the path to your dataset directory
dataset_dir = 'path/to/dataset'

# Set the image dimensions
img_width, img_height = 64, 64

# Set the number of gesture classes
num_classes = 10

# Load and preprocess the dataset
data = []
labels = []

# Load and preprocess the images
for gesture_class in range(num_classes):
    class_dir = os.path.join(dataset_dir, f'gesture{gesture_class}')
    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (img_width, img_height))
        image = img_to_array(image)
        data.append(image)
        labels.append(gesture_class)

# Convert the data and labels to NumPy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize the pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert the labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Define the gesture recognition model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('gesture_recognition_model.h5')
