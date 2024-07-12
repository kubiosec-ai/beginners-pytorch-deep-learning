import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt

# Transfer Learning
transfer_model = models.resnet50(weights=ResNet50_Weights.DEFAULT) 

# Freezing parameters
for name, param in transfer_model.named_parameters():
    if "bn" not in name:
        param.requires_grad = False

# Replacing the classifier
transfer_model.fc = nn.Sequential(
    nn.Linear(transfer_model.fc.in_features, 500),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 2)
) 

# Training function
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, Accuracy = {:.2f}'.format(
            epoch, training_loss, valid_loss, num_correct / num_examples))

# Function to check if image is valid
def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False

# Image transformations
img_transforms = transforms.Compose([
    transforms.Resize((64, 64)),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loading datasets
train_data_path = "../data/train/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=img_transforms, is_valid_file=check_image)
val_data_path = "../data//val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=img_transforms, is_valid_file=check_image)

# Data loaders
batch_size = 64
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(len(val_data_loader.dataset))

transfer_model.to(device)
optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)

# Training the model
train(transfer_model, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=5, device=device)

# Learning rate finder
def find_lr(model, loss_fn, optimizer, train_loader, init_value=1e-8, final_value=10.0, device="cpu"):
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Crash out if loss explodes
        if batch_num > 1 and loss > 4 * best_loss:
            return (log_lrs[10:-5], losses[10:-5]) if len(log_lrs) > 20 else (log_lrs, losses)

        # Record the best loss
        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # Store the values
        losses.append(loss.item())
        log_lrs.append(lr)

        # Do the backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update the lr for the next step and store
        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr

    return (log_lrs[10:-5], losses[10:-5]) if len(log_lrs) > 20 else (log_lrs, losses)

lrs, losses = find_lr(transfer_model, torch.nn.CrossEntropyLoss(), optimizer, train_data_loader, device=device)
plt.plot(lrs, losses)
plt.xscale("log")
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.show()

# Custom Transforms
def _random_colour_space(x):
    return x.convert("HSV")

colour_transform = transforms.Lambda(lambda x: _random_colour_space(x))
random_colour_transform = torchvision.transforms.RandomApply([colour_transform])

class Noise():
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor):
        noise = torch.zeros_like(tensor).normal_(self.mean, self.stddev)
        return tensor.add_(noise)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, stddev={self.stddev})"

custom_transform_pipeline = transforms.Compose([random_colour_transform, Noise(0.1, 0.05)])

# Ensembles
models_ensemble = [models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device), models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)]
predictions = [F.softmax(m(torch.rand(1, 3, 224, 244).to(device)), dim=1) for m in models_ensemble]
avg_prediction = torch.stack(predictions).mean(0).argmax()

print(avg_prediction)
print(torch.stack(predictions))
