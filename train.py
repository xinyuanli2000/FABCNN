import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models 
from torchvision.datasets import ImageFolder
from classes.classifier.UNetHybridFeedbackCNN import UNetRecurrentConcatenatingHybridFeedbackCNN

#from logging_support import log_info, init_logging
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define training parameters
feedback_layers = [0, 5, 10, 19, 28]
num_iterations = [0,1,2]
learning_rate = 0.001
momentum = 0.9
num_epochs = 100
batch_size = 64

# Set up data transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ImageNet dataset
train_dataset = ImageFolder(root="/nobackup/sc22x2l/feedback_attention/train", transform=train_transform)
val_dataset = ImageFolder(root="/nobackup/sc22x2l/feedback_attention/test", transform=val_transform)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

vgg19 = models.vgg19(pretrained=True)
    #vgg19.load_state_dict(torch.load("path_to_pretrained_vgg19_weights"))
vgg19.to(device)

insertion_layers = ','.join(str(layer) for layer in feedback_layers)
num_iterations=2
        # Create the UNetRecurrentConcatenatingHybridFeedbackCNN model
model = UNetRecurrentConcatenatingHybridFeedbackCNN(vgg19, feedback_module_type="convolution",
                                                            insertion_layers=insertion_layers,
                                                            device=device,num_iterations=2)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
scheduler = StepLR(optimizer, step_size=30, gamma=0.7)

# Training loop
best_acc = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    epochs = epoch
    # Update learning rate
    scheduler.step()
    with tqdm(total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{num_epochs}",
                    unit="batch") as pbar:
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({"Loss": train_loss / (batch_idx + 1), "Acc": 100. * correct / total})
            pbar.update()

        # Validation loop
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total
    print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%")

# Save the trained model
    if val_acc > best_acc:
      best_acc = val_acc
      model_filename = f"UNetRecurrentConcatenatingHybridFeedbackCNN_Layers-{feedback_layers}_Iterations-{num_iterations}_Epochs-{epochs}.pt"
      torch.save(model.state_dict(), model_filename)
      print(f"Model saved as {model_filename}")
