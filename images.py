import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

# Load PNG images 
image_files = [f for f in os.listdir('leapGestRecog') if f.endswith('.png')] # images
images = []
for f in image_files:
    img = cv2.imread(os.path.join('images',f))
    img = cv2.resize(img, (224, 224))
    images.append(img)
images = np.array(images)

# Define model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Train model on images
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
model.fit(images, labels, epochs=10) 

# Rest of code to predict on webcam frames...
