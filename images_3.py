import cv2
import numpy as np 
import os
import tensorflow as tf

tf.config.run_functions_eagerly(True)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

# Load images
image_files = [f for f in os.listdir('leapGestRecog') if f.endswith('.png')]
images = []
for f in image_files:
    img = cv2.imread(os.path.join('leapGestRecog', f))
    img = cv2.resize(img, (224, 224))
    images.append(img)
images = np.array(images)

# Extract labels from file names
labels = []
for f in image_files:
    label = int(f.split('_')[0]) # Get gesture class from filename
    labels.append(label)
#labels = to_categorical(labels) 
# Encode labels without np.max()
num_classes = 10 
encoded_labels = []
for label in labels:
  encoded = np.zeros(num_classes)
  encoded[label] = 1
  encoded_labels.append(encoded)
encoded_labels = np.array(encoded_labels)

# Define and train model
model = tf.keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
model.fit(images, encoded_labels, epochs=10)

# Save model
model.save('gesture_model.h5') 

# Load saved model
loaded_model = tf.keras.models.load_model('gesture_model.h5')

# Make predictions...
# Predict on webcam frames...


# Load and preprocess images 

# Get labels
#labels = [...]

