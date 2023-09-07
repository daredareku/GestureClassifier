#
import cv2
import torch
import torchvision.transforms as transforms
from GestureRecognitionModelpth import GestureRecognitionModel as GestureClassifier  # Replace 'model' with the actual architecture you used

# Load the pre-trained model
model = GestureClassifier(10)  # Replace 'GestureClassifier' with the actual class name of your model
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

while True:
    # Capture the image from the webcam
    ret, frame = cap.read()

    # Preprocess the image
    image = preprocess(frame)

    # Make a prediction
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        prediction = torch.argmax(output, dim=1).item()

    # Display the prediction
    cv2.putText(frame, str(prediction), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
