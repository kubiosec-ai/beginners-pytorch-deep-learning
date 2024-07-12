import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFile

# Ensure truncated images can be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Image transformations
img_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define the network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
simplenet = SimpleNet()

# Check for GPU availability and move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simplenet.to(device)

# Load the model state for inference
simplenet.load_state_dict(torch.load("../model/simplenet"))

# Making predictions
labels = ['cat', 'fish']

# Load and transform the image
img = Image.open("../data/val/cat/33524095_c9573d494e.jpg")
img = img_transforms(img).to(device)
img = torch.unsqueeze(img, 0)

# Set the model to evaluation mode and make a prediction
simplenet.eval()
with torch.no_grad():  # Disable gradient computation for inference
    prediction = F.softmax(simplenet(img), dim=1)
    prediction = prediction.argmax()
print(labels[prediction])
