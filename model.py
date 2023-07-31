#
import torch
import torch.nn as nn
import torchvision.models as models

class GestureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.fc = nn.Linear(self.cnn.fc.in_features, 10)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
