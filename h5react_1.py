#
import cv2
import torch
import torchvision.transforms as transforms

# Map of prediction class to gesture name
gesture_map = {
    0: 'Palm',
    1: 'L',
    2: 'Fist',
    3: 'FistMoved',
    4: 'Thumb',
    5: 'Index',
    6: 'Okay',
    7: 'PalmMoved',
    8: 'C',
    9: 'Down'
}

# Load the model
model = torch.load('gesture_recognition_model.h5', map_location='cpu')
model.eval()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame
    ret, frame = cap.read()

    # Check if frame is empty
    if not ret:
        break

    # Preprocess the frame
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Resize frame
    if frame.size != (0, 0):
        frame = cv2.resize(frame, (64, 64))

    input_tensor = transform(frame).unsqueeze(0)

    # Classify gesture
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()

    # Get gesture name
    gesture = gesture_map.get(pred, "Undefined")

    # Display gesture
    cv2.putText(frame, gesture, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
