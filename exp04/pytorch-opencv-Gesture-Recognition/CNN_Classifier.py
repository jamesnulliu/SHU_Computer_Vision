
import dataReader
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

CLASSIFIER_ROOT_ORIGIN = "CNN_Classifier_origin"
CLASSIFIER_ROOT_EXTEND = "CNN_Classifier_extend"
os.makedirs(CLASSIFIER_ROOT_ORIGIN, exist_ok=True)
os.makedirs(CLASSIFIER_ROOT_EXTEND, exist_ok=True)

ngpu = 1
batch_size = 128
whether_fake = 0 
LR = 0.001

torch.cuda.empty_cache()
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def processTrainAndValidation_cpu():
    # data = h5py.File("data.h5", "r")
    # Each img is 64*64, 3 channels
    # imgs = np.array(data['X'])
    # labels = np.array(data['Y'])
    imgs, labels = dataReader.getImagesAndLabels("trainImgs")

    imgs = np.transpose(imgs, (0, 3, 1, 2))
    # Normalize imgs
    imgs = imgs / 255.0

    imgs = torch.tensor(imgs).type(torch.float16)
    labels = torch.tensor(labels)
    print("Memory taken of images (MB): ", imgs.element_size() * imgs.nelement() / 1024 / 1024) 

    dataset = torch.utils.data.TensorDataset(imgs, labels)

    # Split train and validation and test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    if whether_fake:
        fakeImgs, fakeLabels = dataReader.getImagesAndLabels("fakeImgs")
        fakeImgs = np.array(fakeImgs, dtype=np.float16)
        fakeLabels = np.array(fakeLabels)
        # Add fake to train_dataset
        fakeImgs = np.transpose(fakeImgs, (0, 3, 1, 2))
        fakeImgs = fakeImgs / 255.0
        fakeImgs = torch.tensor(fakeImgs).type(torch.float16)
        fakeLabels = torch.tensor(fakeLabels)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, torch.utils.data.TensorDataset(fakeImgs, fakeLabels)])


    traindataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    valdataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True
    )
    testdataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    return traindataloader, valdataloader, testdataloader

class CNN_Classifier(torch.nn.Module):
    def __init__(self):
        super(CNN_Classifier, self).__init__()
        self.conv1 = torch.nn.Sequential(
            # Input size: 3*64*64
            torch.nn.Conv2d(3, 16, 5, 1, 2),
            # Output size: 16*64*64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
            # Output size: 16*32*32
        )
        self.conv2 = torch.nn.Sequential(
            # Input size: 16*32*32
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            # Output size: 32*32*32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
            # Output size: 32*16*16
        )
        self.dense = torch.nn.Sequential(
            # Input size: 32*16*16
            torch.nn.Linear(32 * 16 * 16, 64),
            # Output size: 64
            torch.nn.ReLU(),
            torch.nn.Linear(64, 6),
            # Output size: 5
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        res = conv2_out.view(conv2_out.size(0), -1)
        out = self.dense(res)
        return out

def train(model, trainDataloader, valDataloader, optimizer, loss_func, num_epochs=10):
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        model.train()
        avgTrainloss = 0
        for step, (batch_x, batch_y) in enumerate(trainDataloader):
            batch_x = batch_x.to(device)
            batch_x = batch_x.type(torch.float32)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            loss = loss_func(output, batch_y.long())
            avgTrainloss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avgTrainloss /= len(trainDataloader)
        train_loss.append(avgTrainloss)

        model.eval()
        avgValLoss = 0
        for step, (batch_x, batch_y) in enumerate(valDataloader):
            batch_x = batch_x.to(device)
            batch_x = batch_x.type(torch.float32)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            loss = loss_func(output, batch_y.long())
            avgValLoss += loss.item()
        avgValLoss /= len(valDataloader)
        val_loss.append(avgValLoss)
        if whether_fake:
            torch.save(model.state_dict(), CLASSIFIER_ROOT_EXTEND+ "/epoch" + str(epoch) + ".pth")
        else:
            torch.save(model.state_dict(), CLASSIFIER_ROOT_ORIGIN + "/epoch" + str(epoch) + ".pth")
    return train_loss, val_loss

def getClassifier(path):
    model = CNN_Classifier()
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    trainDataloader, valDataloader, testDataloader = processTrainAndValidation_cpu()
    model = CNN_Classifier()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()
    train_loss, val_loss = train(model, trainDataloader, valDataloader, optimizer, loss_func, num_epochs=10)
    # Plot train loss and validation loss
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="validation loss")
    plt.legend()
    plt.show()

    # Test
    model.eval()
    test_loss = 0
    correct = 0
    for step, (batch_x, batch_y) in enumerate(testDataloader):
        batch_x = batch_x.to(device)
        batch_x = batch_x.type(torch.float32)
        batch_y = batch_y.to(device)
        output = model(batch_x)
        test_loss += loss_func(output, batch_y.long()).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(batch_y.view_as(pred)).sum().item()
    test_loss /= len(testDataloader)
    print("Test loss: ", test_loss)
    print("Test accuracy: ", correct / len(testDataloader.dataset))
    
