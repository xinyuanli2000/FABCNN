import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision import models
from classes.classifier.UNetHybridFeedbackCNN import UNetRecurrentConcatenatingHybridFeedbackCNN
from torchvision.datasets import ImageFolder
# Set device
device = torch.device("cpu")
import numpy as np

def top_k_accuracy(predictions, targets, k=5):
    assert len(predictions) == len(targets), "预测结果和标签数量不一致"
    num_samples = len(targets)
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:] 
    correct = [1 if targets[i] in top_k_preds[i] else 0 for i in range(num_samples)]
    top_k_acc = sum(correct) / num_samples
    return top_k_acc

# Define feedback layers and number of iterations (make sure to use the same values as in the training script)
feedback_layers = [0, 5, 10, 19, 28]
num_iterations = 1

# Set up data transformations for the test dataset
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test ImageNet dataset
test_dataset = ImageFolder(root="D:/study/msc/project/feedback-attention-cnn-main/dataset/Test_name", transform=test_transform)

# Create the data loader for the test dataset
batch_size = 8
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("data load")
# Load the pre-trained VGG19 model
vgg19 = models.vgg19(pretrained=True)
vgg19.to(device)

# Create the UNetRecurrentConcatenatingHybridFeedbackCNN model
insertion_layers = ','.join(str(layer) for layer in feedback_layers)
model = UNetRecurrentConcatenatingHybridFeedbackCNN(vgg19, feedback_module_type="convolution",
                                                   insertion_layers=insertion_layers,
                                                   device=device, num_iterations=num_iterations)
model.to(device)

# Load the saved model weights
model_filename = f"D:/study/msc/project/feedback-attention-cnn-main/UNetRecurrentConcatenatingHybridFeedbackCNN_Layers-[0, 5, 10, 19, 28]_Iterations-1_Epochs-45.pt"
model.load_state_dict(torch.load(model_filename))
model.eval()
print("model load")
# Evaluate the model on the test dataset
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
test_correct = 0
test_total = 0
all_predictions = []
all_targets = []
num_classes = 100
test_correct_per_class = {i:0 for i in range(num_classes)}
test_total_per_class = {i: 50 for i in range(num_classes)}
test_accuracy_per_class = {}
# 混淆矩阵
confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        # 更新每个类别的正确预测数
        for i in range(targets.shape[0]):
            label = targets[i].item() 
            test_correct_per_class[label] += predicted[i] == label
            pred= predicted[i]
            confusion_matrix[label, pred] += 1 
    for i in range(num_classes):
        test_accuracy_per_class[i] = 100. * test_correct_per_class[i] / test_total_per_class[i]
              
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
test_top_5_accuracy = 100. * top_k_accuracy(all_predictions, all_targets, k=5)

test_loss /= len(test_loader)
test_accuracy = 100. * test_correct / test_total

print(f"Test Loss: {test_loss:.4f} | Test Accuracy (Top-1): {test_accuracy:.2f}% | Test Accuracy (Top-5): {test_top_5_accuracy:.2f}%")
print(test_accuracy_per_class)
print(confusion_matrix.numpu())