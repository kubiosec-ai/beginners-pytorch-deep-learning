import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image

# Update the model initialization to use the 'weights' parameter
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

resnet50 = torch.hub.load('pytorch/vision', 'resnet50', weights='DEFAULT')

print(alexnet)
print(resnet50)

