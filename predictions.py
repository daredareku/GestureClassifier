#
import cv2
import torch
import torchvision

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

# Load model 
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 10) # 10 gesture classes

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame
    ret, frame = cap.read()
    
    # Preprocess
    img = torchvision.transforms.ToTensor()(frame)
    
    # Classify gesture
    output = model(img.unsqueeze(0)) 
    pred = output.argmax(dim=1).item()
    
    # Get gesture name
    # Get gesture name  
    gesture = gesture_map.get(pred, "Undefined")
    '''if pred == 0: 
        gesture = 'Palm'
    elif pred == 1:
        gesture = 'L'
    '''
        
    # Display 
    cv2.putText(frame, gesture, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
