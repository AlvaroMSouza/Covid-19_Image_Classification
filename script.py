import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Precision, Recall
from torchsummary import summary


from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((128,128))])

train_dataset = ImageFolder('xray_dataset_covid19/train', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

img, label = next(iter(train_loader))
print(img.shape, label.shape)

img = img[0].permute(1,2,0) 

# show image
plt.imshow(img)
plt.show()


# Convolutional Neural Network

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor= nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.classifier = nn.Linear(64*32*32, num_classes)
        
    def forward(self, x):
        x= self.feature_extractor(x)
        x = self.classifier(x) 
        return x





num_classes = 2
model = CNN(num_classes = num_classes)
summary(model, (3, 128, 128))

# Training the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = optim.adam(model.parameters(), lr=0.001)

accuracy = Accuracy()

# Train epochs

