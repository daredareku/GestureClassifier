#
import cv2
import torch
import torchvision
from torchvision import datasets, transforms

# Load training data 
train_data = datasets.ImageFolder('leapGestRecog', transform=transforms.ToTensor()) 

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
#labels = to_categorical(labels) 
# Encode labels without np.max()
num_classes = 10 
encoded_labels = []
for label in labels:
  encoded = np.zeros(num_classes)
  encoded[label] = 1
  encoded_labels.append(encoded)
encoded_labels = np.array(encoded_labels)


# Train model
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, len(train_data.classes))

# Train model 
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
  for img, label in train_data:
    optimizer.zero_grad()
    output = model(img)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

# Load gesture name mapping
gesture_map = {idx:name for idx, name in enumerate(train_data.classes)}

# Webcam inference loop
cap = cv2.VideoCapture(0) 

while True:

  # Read frame
  ret, frame = cap.read()
  
  # Classify
  img = transform(frame) # Preprocess
  output = model(img)
  pred = output.argmax()
  
  # Get gesture
  gesture = gesture_map[pred]
  
  # Display
  cv2.putText(frame, gesture, ...)
  cv2.imshow('Gestures', frame)

  if cv2.waitKey(1) == 27:
    break
      
cap.release()
cv2.destroyAllWindows()
