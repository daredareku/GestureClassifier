import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
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
                if filename.endswith('.jpg'):
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

# Change directory to the specified root_dir
root_dir = r'E:\1\Downloads\1\Downloads\2023\Geekbrains\DIPLOMA\GestureClassifier\leapGestRecog'
os.chdir(root_dir)

# Create the dataset instance
dataset = GestureDataset('.')
