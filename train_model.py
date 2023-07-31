#
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import GestureDataset # Your PyTorch dataset

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up the training parameters
batch_size = 32
learning_rate = 1e-3
num_epochs = 10

# Set up the data loaders
train_dataset = GestureDataset('leapGestRecog/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = GestureDataset('leapGestRecog/val')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Set up the model
model = GestureClassifier().to(device)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    # Train for one epoch
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_acc += (outputs.argmax(dim=1) == labels).sum().item()

    train_loss /= len(train_dataset)
    train_acc /= len(train_dataset)

    # Evaluate on the validation set
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_acc += (outputs.argmax(dim=1) == labels).sum().item()

    val_loss /= len(val_dataset)
    val_acc /= len(val_dataset)

    # Print the training and validation metrics
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Save the model
torch.save(model.state_dict(), 'model.pth')
