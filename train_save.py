#
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from GestureDataset1 import GestureDataset  # Assuming you have the GestureDataset class in a separate file

# Set the necessary parameters
root_dir = r'E:\1\Downloads\1\Downloads\2023\Geekbrains\DIPLOMA\GestureClassifier\leapGestRecog\leapGestRecog'
# '.' # 'path/to/dataset/root'  # Path to the root directory of your gesture dataset
num_classes = 10  # Number of gesture classes
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Create the dataset and dataloader
dataset = GestureDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the model
model = resnet50(pretrained=False)
model.fc = nn.Linear(2048, num_classes)  # Replace the last fully connected layer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

'''
Traceback (most recent call last):
  File "E:\1\Downloads\1\Downloads\2023\Geekbrains\DIPLOMA\GestureClassifier\train_save.py", line 20, in <module>
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  File "C:\Users\vital.DESKTOP-VE61DMS\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\utils\data\dataloader.py", line 351, in __init__
    sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
  File "C:\Users\vital.DESKTOP-VE61DMS\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\utils\data\sampler.py", line 107, in __init__
    raise ValueError("num_samples should be a positive integer "
ValueError: num_samples should be a positive integer value, but got num_samples=0
'''
