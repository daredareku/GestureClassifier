#
import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import GestureClassifier
from dataset import GestureDataset

# Set the device for running the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Set the root directory of the dataset
root_dir = 'leapGestRecog'

# Define the image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create the training and validation datasets
temp=os.path.join(root_dir, 'train')
train_dataset = GestureDataset(temp, preprocess)
val_dataset = GestureDataset(os.path.join(root_dir, 'val'), preprocess)

# Create the training and validation data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Create an instance of the model
model = GestureClassifier().to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    # Train the model for one epoch
    model.train()
    train_loss = 0.0
    train_corrects = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_corrects += torch.sum(preds == labels.data)

    train_loss = train_loss / len(train_dataset)
    train_acc = train_corrects.double() / len(train_dataset)

    # Evaluate the model for one epoch
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        val_loss += loss.item() * images.size(0)
        val_corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)

    print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    print()

# Save the trained model to a file
torch.save(model.state_dict(), 'model.pth')

# Load the pre-trained model for inference
model = GestureClassifier().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Set up the webcam
cap = cv2.VideoCapture(0)

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Capture the image from the webcam
    ret, frame = cap.read()

    # Detect faces in the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectedMultiScale(gray, 1.3, 5)

    # Preprocess the image
    image = preprocess(frame).to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        prediction = torch.argmax(output, dim=1).item()

    # Display the prediction and the face detection results
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, str(prediction), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
