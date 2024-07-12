import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import json
import requests

# Update the model initialization to use the 'weights' parameter
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
resnet50 = torch.hub.load('pytorch/vision', 'resnet50', weights='DEFAULT')

# Set the models to evaluation mode
alexnet.eval()
resnet50.eval()

# Define the image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
img_path = "../data/val/cat/33524095_c9573d494e.jpg"
img = Image.open(img_path)

# Preprocess the image
img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

# Classify the image using AlexNet
with torch.no_grad():
    alexnet_output = alexnet(img_tensor)
    resnet50_output = resnet50(img_tensor)

# Get the predicted class index for AlexNet
alexnet_probabilities = torch.nn.functional.softmax(alexnet_output[0], dim=0)
alexnet_top5_prob, alexnet_top5_catid = torch.topk(alexnet_probabilities, 5)

# Get the predicted class index for ResNet50
resnet50_probabilities = torch.nn.functional.softmax(resnet50_output[0], dim=0)
resnet50_top5_prob, resnet50_top5_catid = torch.topk(resnet50_probabilities, 5)

# Load the labels for ImageNet classes
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels_map = requests.get(LABELS_URL).json()

# Print the top 5 predictions for AlexNet
print("AlexNet Predictions:")
for i in range(5):
    print(f"{labels_map[alexnet_top5_catid[i]]}: {alexnet_top5_prob[i].item()}")

# Print the top 5 predictions for ResNet50
print("ResNet50 Predictions:")
for i in range(5):
    print(f"{labels_map[resnet50_top5_catid[i]]}: {resnet50_top5_prob[i].item()}")

