import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Precision, Recall
from torchsummary import summary
from torchvision.models import resnet50, ResNet50_Weights

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the dataset
train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomAutocontrast(),
                                       transforms.ToTensor(),
                                      transforms.Resize((128,128))])

test_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((128,128))])

train_dataset = ImageFolder('xray_dataset_covid19/train', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ImageFolder('xray_dataset_covid19/test', transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32)

img, label = next(iter(train_loader))
print(img.shape, label.shape)

img = img[0].permute(1,2,0) 

# show image
plt.imshow(img)
plt.show()

# Print how many images are in the dataset 
print(f'Train dataset: {len(train_dataset)} images')
print(f'Test dataset: {len(test_dataset)} images')

# Print the number of classes
print(f'Classes: {train_dataset.classes}')

# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor= nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        self.classifier = nn.Linear(128*32*32, num_classes)
        
    def forward(self, x):
        x= self.feature_extractor(x)
        x = self.classifier(x) 
        return x

num_classes = 2
model = CNN(num_classes = num_classes).to(device)
summary(model, (3, 128, 128))

# Training the model

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

# Train epochs
for epoch in range(20):
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'Models/model_CNN.pth')

# Testing the model
metric_acc = Accuracy(task='binary', num_classes=num_classes, average='macro').to(device)
metric_precision = Precision(task='binary', num_classes=num_classes, average='macro').to(device)
metric_recall = Recall(task='binary', num_classes=num_classes, average='macro').to(device)

model.eval()
with torch.no_grad():
    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        metric_acc(predicted, label)
        metric_precision(predicted, label)
        metric_recall(predicted, label)
    
acc = metric_acc.compute()   
prec = metric_precision.compute()
rec = metric_recall.compute()
print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}')


# Resnet model
resnet = resnet50(pretrained= True).to(device)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes).to(device)

# Train the model

optimizer = optim.Adam(resnet.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train epochs
for epoch in range(20):
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = resnet(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
        
torch.save(resnet.state_dict(), 'Models/model_Resnet.pth')
        
# Testing the model
metric_acc_resnet = Accuracy(task='binary', num_classes=2, average='macro').to(device)
metric_precision_resnet = Precision(task='binary', num_classes=2, average='macro').to(device)
metric_recall_resnet = Recall(task='binary', num_classes=2, average='macro').to(device)

resnet.eval()
with torch.no_grad():
    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        output = resnet(image)
        _, predicted = torch.max(output, 1)
        metric_acc_resnet(predicted, label)
        metric_precision_resnet(predicted, label)
        metric_recall_resnet(predicted, label)
    
acc_resnet = metric_acc_resnet.compute()   
prec_resnet = metric_precision_resnet.compute()
rec_resnet = metric_recall_resnet.compute()
print(f'Accuracy: {acc_resnet}, Precision: {prec_resnet}, Recall: {rec_resnet}')


# Move metrics to CPU and convert to numpy
metrics = {
    'Model': ['CNN', 'ResNet'],
    'Accuracy': [acc.cpu().numpy().item(), acc_resnet.cpu().numpy().item()],
    'Precision': [prec.cpu().numpy().item(), prec_resnet.cpu().numpy().item()],
    'Recall': [rec.cpu().numpy().item(), rec_resnet.cpu().numpy().item()]
}

df = pd.DataFrame(metrics)
df = pd.melt(df, id_vars='Model', var_name='Metrics', value_name='Values')

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Values', hue='Metrics', data=df)
plt.title('Comparison of CNN and ResNet Models')
plt.xlabel('Model')
plt.ylabel('Metric Values')
plt.legend(loc='upper left')
plt.show()



