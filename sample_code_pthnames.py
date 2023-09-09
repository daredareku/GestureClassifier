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

try:
    print('tensorflow '+tf.__version__)
    print('torch '+torch.__version__)
    print('Press Ctrl-C for exit:')
    while True:
        # Capture the image from the webcam
        ret, frame = cap.read()

        # Preprocess the image
        image = preprocess(frame)

        # Make a prediction
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            prediction = torch.argmax(output, dim=1).item()

        # Get gesture name
        gesture = gesture_map.get(prediction, "Undefined")

        # Display the prediction
        cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
