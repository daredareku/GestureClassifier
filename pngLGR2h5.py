#
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array

# Map of prediction class to gesture name 
gesture_map = {
    0: 'Palm',
    1: 'L',
    2: 'Fist',
    3: 'Fist_Moved', 
    4: 'Thumb',
    5: 'Index',
    6: 'Ok',
    7: 'Palm_Moved',
    8: 'C',
    9: 'Down'
}

# Set the path to the LeapGestRecog dataset directory
dataset_dir = 'leapGestRecog' # path to

# Set the image dimensions
img_width, img_height = 64, 64

# Set the number of gesture classes
num_classes = 10

# Load and preprocess the dataset
data = []
labels = []

def lowcase(input_string):
    return input_string.lower()

# Load and preprocess the images
for gesture_class in range(num_classes):
    for j in range(1, 11):
        gm = lowcase(gesture_map[j-1])
        if (j==10):
            class_dir = os.path.join( os.path.join(dataset_dir, '0'+str(gesture_class)) , str(j)+'_'+gm ) # plus  _* in filename
        else:
            class_dir = os.path.join( os.path.join(dataset_dir, '0'+str(gesture_class)) , '0'+str(j)+'_'+gm ) # plus  _* in filename
        print(class_dir)
        for i in range(1, 201):  # Assumes 200 / 1100 images per gesture class
            if (j==10):
                image_file = f'frame_{gesture_class:02d}_10_{i:04d}.png'
            else:
                image_file = f'frame_{gesture_class:02d}_{str(j).zfill(2)}_{i:04d}.png'
            image_path = os.path.join(class_dir, image_file)
            image = cv2.imread(image_path)
            try:
                image = cv2.resize(image, (img_width, img_height))
                image = img_to_array(image)
                data.append(image)
                labels.append(gesture_class)
            except Exception as e:
                if (i==1):
                    print(f"Error processing image: {e}")    
                    print(image_file)

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
print('Saving .h5 file...')
model.save('gesture_recognition_model.h5')
