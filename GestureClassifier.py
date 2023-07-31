#
import torch
import torch.nn as nn
import torch.optim as optim
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

''' In this code, we define a PyTorch module called GestureClassifier that consists of a pre-trained ResNet-18 convolutional neural network (CNN) and a fully connected layer.
 The pre-trained ResNet-18 CNN is used to extract features from the input image, and the fully connected layer is used to classify the hand gesture.

The forward method takes an input tensor x and passes it through the CNN and the fully connected layer. The output is a tensor of size (batch_size, 10), where 10 
is the number of hand gestures in the LeapGestRecog dataset.
'''