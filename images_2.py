import cv2
import numpy as np 
import os
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
labels = to_categorical(labels) 

# Define and train model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(images, labels, epochs=10)

# Predict on webcam frames...
