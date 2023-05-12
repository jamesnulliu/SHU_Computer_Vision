# Build a CNN model and train the classifier with pytorch. The input imgs are 60x60 grayscale imgs.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
from tqdm import tqdm
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from utilities import LRScheduler, EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
args = vars(parser.parse_args())

def process_data(imgs, labels = None):
    """
    Process imgs and labels to torch tensors. Normalize imgs.
    """
    imgs = torch.from_numpy(imgs)
    imgs = imgs.float()
    imgs = imgs / 255.0 * 2 - 1  # Normalization
    imgs = imgs.view(-1, 1, 64, 64)
    imgs = imgs.to(device)
    if labels is None:
        return Variable(imgs)
    labels = torch.from_numpy(labels)
    labels = labels.long()
    labels = labels.to(device)
    return Variable(imgs), Variable(labels)

class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        # Fully connected layers
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_classifier_CNN(train_data, train_label, validation_data, validation_label):
    # Hyper parameters
    num_epochs = 10 
    batch_size = 32 
    learning_rate = 0.001
    
    # Model
    model = CNN_Model().to(device)
    # Print model's information

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_data, train_label = process_data(train_data, train_label)
    train_data = torch.utils.data.TensorDataset(train_data, train_label)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data, validation_label = process_data(validation_data, validation_label)
    validation_data = torch.utils.data.TensorDataset(validation_data, validation_label)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)


    # Utilities
    lr_scheduler, early_stopping, loss_plot_name, acc_plot_name, model_name = getUtilities(optimizer)
 
    # Training
    # lists to store per-epoch loss and accuracy values
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    start = time.time()
    print('============================================================')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")
        print('------------------------------------------------------------')
        train_epoch_loss, train_epoch_accuracy = fit(model, train_data_loader, train_data, optimizer, criterion)
        val_epoch_loss, val_epoch_accuracy = validate(model, validation_data_loader, validation_data, criterion)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        if args['lr_scheduler']:
            lr_scheduler(val_epoch_loss)
        if args['early_stopping']:
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                break
        print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')
        print('============================================================')
    end = time.time()
    # print(f"Training time: {(end-start)/60:.3f} minutes")
    print(f"Training time: {(end-start)} seconds")

    # Plot train_loss and val_loss
    plt.figure(1, edgecolor='black')
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(loss_plot_name)
    plt.legend()

    # Plot train_accuracy and val_accuracy
    plt.figure(2)
    plt.plot(train_accuracy, label='train_accuracy')
    plt.plot(val_accuracy, label='val_accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(acc_plot_name)
    plt.legend()

    return model

def getUtilities(optimizer):
    lr_scheduler = None
    early_stopping = None
    loss_plot_name = 'Loss'
    acc_plot_name = 'Accuracy'
    model_name = 'Model'

    if args['lr_scheduler']:
        print('INFO: Initializing learning rate scheduler')
        lr_scheduler = LRScheduler(optimizer)
        # change the accuracy, loss plot names and model name
        loss_plot_name = 'LRS_Loss Figure'
        acc_plot_name = 'LRS_Accuracy Figure'
        model_name = 'LRS_Model'

    if args['early_stopping']:
        print('INFO: Initializing early stopping')
        early_stopping = EarlyStopping()
        # change the accuracy, loss plot names and model name
        loss_plot_name = 'ES_Loss Figure'
        acc_plot_name = 'ES_Accuracy Figure'
        model_name = 'ES_Model'

    return lr_scheduler, early_stopping, loss_plot_name, acc_plot_name, model_name

def fit(model, train_data_loader, train_dataset, optimizer, criterion):
    print('* Training ...')
    model.train()                                                   # set the model to training mode
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_data_loader), 
                    total=int(len(train_dataset)/train_data_loader.batch_size))

    for i, data in prog_bar:
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        total += target.size(0)
        optimizer.zero_grad()                                       # clear gradients
        outputs = model(data)                                       # forward pass: predict outputs of the data using the model
        loss = criterion(outputs, target)                           # calculate the loss
        train_running_loss += loss.item()                           # update running loss
        _, preds = torch.max(outputs.data, 1)                       # get the predicted class from the maximum value in the output-list of class scores
        train_running_correct += (preds == target).sum().item()     # update running correct count
        loss.backward()                                             # backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()                                            # perform a single optimization step (parameter update)
        
    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / total
    return train_loss, train_accuracy

# validation function
def validate(model, test_data_loader, test_data, criterion):
    print('* Validating')
    model.eval()                                                   # set the model to evaluation mode
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_data_loader), total=int(len(test_data)/test_data_loader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            total += target.size(0)
            outputs = model(data)                                  # forward pass: predict outputs of the data using the model
            loss = criterion(outputs, target)                      # calculate the loss
            val_running_loss += loss.item()                        # update running loss
            _, preds = torch.max(outputs.data, 1)                  # get the predicted class from the maximum value in the output-list of class scores
            val_running_correct += (preds == target).sum().item()  # update running correct count
        
        val_loss = val_running_loss / counter
        val_accuracy = 100. * val_running_correct / total
        return val_loss, val_accuracy