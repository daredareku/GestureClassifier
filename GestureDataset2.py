#
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from PIL import Image

class GestureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.samples = self._make_dataset()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _make_dataset(self):
        samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.png'):
                    path = os.path.join(class_dir, filename)
                    sample = (path, self.class_to_idx[class_name])
                    samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, label

# Set the necessary parameters
root_dir = r'E:\1\Downloads\1\Downloads\2023\Geekbrains\DIPLOMA\GestureClassifier\leapGestRecog'
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
