#
import cv2
import torch
import torchvision
from torchvision import transforms

# Gesture mapping
gesture_map = {0:'Palm', 1:'L', 2:'Fist', 3:'FistMoved', 4:'Thumb', 5:'Index', 
               6:'Okay', 7:'PalmMoved', 8:'C', 9:'Down'}

# Model
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, len(gesture_map))

# Load model weights 
model.load_state_dict(torch.load('leapGestRecog/00')) #model.pth'))

# Transformer 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Webcam 
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Classify gesture
    img = transform(frame)
    out = model(img.unsqueeze(0))
    pred = out.argmax(1).item()
    
    # Get gesture
    gesture = gesture_map[pred]
    
    # Display 
    cv2.putText(frame, gesture, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,  
                1, (0,255,0), 2) 
    cv2.imshow('Gestures', frame)

    if cv2.waitKey(1) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
