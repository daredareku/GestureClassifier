import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('gesture_model.h5') 

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from webcam 
    _, frame = cap.read()
    
    # Preprocess frame
    frame = cv2.resize(frame, (224,224))
    frame = np.expand_dims(frame, axis=0)
    
    # Make prediction
    pred = model.predict(frame)
    gesture_id = np.argmax(pred[0])
    
    # Display predicted gesture name 
    if gesture_id == 0:
        gesture = 'Palm'
    elif gesture_id == 1:
        gesture = 'L'
    # Add remaining gestures
    
    # Display the predicted gesture on frame
    cv2.putText(frame, gesture, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,255,0), 2, cv2.LINE_AA)
    
    # Display webcam frame
    cv2.imshow('Frame', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release webcam
cap.release()
cv2.destroyAllWindows()
