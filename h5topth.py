#
import torch
import tensorflow as tf
import torch.nn as nn

# Set the number of gesture classes
num_classes = 10

# Define the PyTorch model
class GestureRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(GestureRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load the trained TensorFlow model
tf_model = tf.keras.models.load_model('gesture_recognition_model.h5')

# Create an instance of the PyTorch model
pytorch_model = GestureRecognitionModel(num_classes)

# Transfer the weights from the TensorFlow model to the PyTorch model
for tf_layer, torch_layer in zip(tf_model.layers, pytorch_model.modules()):
    if isinstance(torch_layer, nn.Conv2d) or isinstance(torch_layer, nn.Linear):
        torch_layer.weight.data = torch.from_numpy(tf_layer.get_weights()[0])
        torch_layer.bias.data = torch.from_numpy(tf_layer.get_weights()[1])

# Save the PyTorch model
torch.save(pytorch_model.state_dict(), 'gesture_recognition_model.pth')
