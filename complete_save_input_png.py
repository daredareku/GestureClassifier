#
import os
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

# Specify the dataset directory path
data_dir = 'leapGestRecog/00/'

# Check if the dataset directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")

# Define the transformation for preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# Load the dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

# Check if the train and val directories exist
for dataset_type in ['train', 'val']:
    if len(image_datasets[dataset_type]) == 0:
        raise FileNotFoundError(f"No images found in '{os.path.join(data_dir, dataset_type)}' directory.")

# Create data loaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}

# Get the number of classes
num_classes = len(image_datasets['train'].classes)

# Print the number of classes
print("Number of classes:", num_classes)

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the fully connected layer to match the number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(image_datasets['train'])
    epoch_acc = running_corrects.double() / len(image_datasets['train'])

    print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')

# Save the trained model weights
torch.save(model.state_dict(), 'model.pth')
