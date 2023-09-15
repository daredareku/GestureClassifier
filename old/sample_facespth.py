#
import cv2
import tensorflow as tf
import torch
import torchvision.transforms as transforms
from GestureRecognitionModelpth1 import GestureRecognitionModel as GestureClassifier

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

# Set the number of gesture classes
num_classes = 10

# Load the pre-trained model
model = GestureClassifier(num_classes)
model.load_state_dict(torch.load('gesture_recognition_model.pth'))
model.eval()

# Set up the webcam
cap = cv2.VideoCapture(0)

# Define the image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    print('tensorflow '+tf.__version__)
    print('torch '+torch.__version__)

    print('Press Ctrl-C to exit:')
    while True:
        # Capture the image from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Preprocess the face region
            face_image = frame[y:y+h, x:x+w]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image = preprocess(face_image)

            # Make a prediction for the face region
            with torch.no_grad():
                output = model(face_image.unsqueeze(0))
                prediction = torch.argmax(output, dim=1).item()

            # Get gesture name
            gesture = gesture_map.get(prediction, "Undefined")

            # Display the gesture prediction
            cv2.putText(frame, gesture, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Exit on 'q' key press or Ctrl-C
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # 'q' key or Esc key
            break
except KeyboardInterrupt:
    pass
    
# Clean up
cap.release()
cv2.destroyAllWindows()
