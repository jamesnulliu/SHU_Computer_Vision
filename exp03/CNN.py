# Build a CNN model and train the classifier with pytorch. The input imgs are 60x60 grayscale imgs.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_data(imgs, labels = None):
    """
    Process imgs and labels to torch tensors. Normalize imgs.
    """
    imgs = torch.from_numpy(imgs)
    imgs = imgs.float()
    imgs = imgs / 255.0 * 2 - 1  # Normalization
    imgs = imgs.view(-1, 1, 60, 60)
    imgs = imgs.to(device)
    if labels is None:
        return Variable(imgs)
    labels = torch.from_numpy(labels)
    labels = labels.long()
    labels = labels.to(device)
    return Variable(imgs), Variable(labels)

class CNN_Model(torch.nn.Module):
    """
    1 batch, 64 imgs, 60x60 pixels.
    """
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        # Fully connected layers
        x = x.view(-1, 32 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_classifier_CNN(train_data, train_label):
    # Hyper parameters
    num_epochs = 10 
    batch_size = 32 
    learning_rate = 0.001

    train_data, train_label = process_data(train_data, train_label)
    train_data = torch.utils.data.TensorDataset(train_data, train_label)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Model
    model = CNN_Model()
    model.cuda()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0.0
        # print("Epoch {}/{}".format(epoch, num_epochs))
        # print("--------------------------------------------")
        for img, label in train_data_loader:
            # img = img.unsqueeze(1)``
            outputs = model(img)
            _, pred = torch.max(outputs.data, 1)

            # print(outputs.type)
            loss = criterion(outputs, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            train_correct += torch.sum(pred == label.data)

        # Print train loss in this epoch
        # print("Loss is:{:.4f}".format(train_loss / len(train_data)))

    train_loss_avg = train_loss / len(train_data)
    train_acc = 100 * train_correct / len(train_data)
    # print("--------------------------------------------")
    # print("Final Loss is:{:.4f}, Train Accuracy is:{:.4f}%".format(train_loss_avg, train_acc))
    # print("--------------------------------------------")

    torch.save(model.state_dict(), 'cnn.pkl')

    return model