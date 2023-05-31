import argparse
import torch
import torch.nn as nn
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import h5py
from IPython.display import HTML

import dataReader

os.makedirs("fakeImgs/0", exist_ok=True)
os.makedirs("fakeImgs/1", exist_ok=True)
os.makedirs("fakeImgs/2", exist_ok=True)
os.makedirs("fakeImgs/4", exist_ok=True)
os.makedirs("fakeImgs/5", exist_ok=True)

GAN_MODEL_PATH = "GAN_Model"

# Set random seed for reproducibility
manualSeed = random.randint(1, 10000)  # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Batch size during training
batch_size = 128
# Number of channels in the training images. For color images this is 3
num_channels = 3
# Size of z latent vector (i.e. size of generator input)
num_ZLatent = 100
# Size of feature maps in generator
num_GFeature = 75
# Size of feature maps in discriminator
num_DFeature = 70
# Number of training epochs
num_epochs = 100
# Learning rate for optimizers
learning_rate = 0.000085
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.65
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Decide which device we want to run on
myGPU = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
myCPU = torch.device("cpu")
# Choose the number gesture to generate
chosen_number = 0


def processData2TensorCPU(chosen_number=None):
    imgs, labels = dataReader.getImagesAndLabels(dataReader.TRAIN_ROOT)
    imgs = np.array(imgs)
    labels = np.array(labels)
    # Get all imgs that labels are {choosenNumber}
    if chosen_number is not None:
        imgs = imgs[labels == chosen_number]
        labels = labels[labels == chosen_number]
    # Covert each img from (64,64,3) to (3,64,64)
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    # Normalize imgs
    imgs = imgs / 255.0
    # To tensor and to device (torch.float32)
    imgs = torch.tensor(imgs).type(torch.float32)
    labels = torch.tensor(labels).type(torch.float32)
    dataset = torch.utils.data.TensorDataset(imgs, labels)
    # Split to train, valid and test
    trainSize = int(len(dataset) * 0.8)
    validSize = int(len(dataset) * 0.1)
    testSize = len(dataset) - trainSize - validSize
    trainSet, validSet, testSet = torch.utils.data.random_split(
        dataset, [trainSize, validSize, testSize]
    )
    trainDataloader = torch.utils.data.DataLoader(
        trainSet, batch_size=batch_size, shuffle=True
    )
    testDataloader = torch.utils.data.DataLoader(
        testSet, batch_size=batch_size, shuffle=True
    )
    valDataloader = torch.utils.data.DataLoader(
        validSet, batch_size=batch_size, shuffle=True
    )
    return trainDataloader, valDataloader, testDataloader


def plotSomeInDataloader(dataloader):
    # Plot some training images
    # real_batch is a tuple (imgs, labels)
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    imgGrid = (
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(myGPU)[:32], padding=2, normalize=True
            ).cpu(),
            (1, 2, 0),
        ).numpy()
        * 255
    ).astype(np.uint8)
    plt.imshow(imgGrid)
    plt.show()


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(num_ZLatent, num_GFeature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_GFeature * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(num_GFeature * 8, num_GFeature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_GFeature * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(num_GFeature * 4, num_GFeature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_GFeature * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(num_GFeature * 2, num_GFeature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_GFeature),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(num_GFeature, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)


def createGenerator():
    # Create the generator
    netG = Generator(ngpu).to(myGPU)
    # Handle multi-GPU if desired
    if (myGPU.type == "cuda") and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)
    return netG


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(num_channels, num_DFeature, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(num_DFeature, num_DFeature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_DFeature * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(num_DFeature * 2, num_DFeature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_DFeature * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(num_DFeature * 4, num_DFeature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_DFeature * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(num_DFeature * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


def createDiscriminator():
    # Create the Discriminator
    netD = Discriminator(ngpu).to(myGPU)
    # Handle multi-GPU if desired
    if (myGPU.type == "cuda") and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)
    return netD


def train(dataloader):
    netD = createDiscriminator()
    netG = createGenerator()

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, num_ZLatent, 1, 1, device=myGPU)
    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_imgs = data[0].to(myGPU)
            b_size = real_imgs.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=myGPU)
            # Forward pass real batch through D
            output = netD(real_imgs).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, num_ZLatent, 1, 1, device=myGPU)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % ( epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,)
                )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or (
                (epoch == num_epochs - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        if iters > 8000:
            break

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    imgs = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(
        fig, imgs, interval=1000, repeat_delay=1000, blit=True
    )
    plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    realImgGrid = np.transpose(
        vutils.make_grid(real_batch[0].to(myGPU)[:64], padding=5, normalize=True).cpu(),
        (1, 2, 0),
    )

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(realImgGrid)

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    fakeImgGrid = np.transpose(img_list[-1], (1, 2, 0))
    plt.imshow(fakeImgGrid)
    plt.show()

    # Save the model
    torch.save(netG, GAN_MODEL_PATH + "/generator_" + str(chosen_number) + ".pth")
    torch.save(netD, GAN_MODEL_PATH + "/discriminator_" + str(chosen_number) + ".pth")


def loadModel(chosen_number):
    netG = torch.load(GAN_MODEL_PATH + "/generator_" + str(chosen_number) + ".pth")
    netD = torch.load(GAN_MODEL_PATH + "/discriminator_" + str(chosen_number) + ".pth")
    return netG, netD


def generateFake(netG, netD, real_label, numImgs):
    numGenerated = 0
    fakeImgs = []
    while numGenerated < numImgs:
        # Generate fake image batch with G
        noise = torch.randn(numImgs, num_ZLatent, 1, 1, device=myGPU)
        label = torch.full((numImgs,), real_label, device=myGPU)
        fake = netG(noise)
        label.fill_(real_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        fake = fake.detach()
        fake = fake[output > 0.5]
        fakeImgs.extend(fake)
        print(len(fakeImgs))
        numGenerated += len(fake)
    return fakeImgs


if __name__ == "__main__":
    for chosen_number in [0, 1]:
        print(chosen_number)
        trainDataloader,_,_ = processData2TensorCPU(chosen_number=chosen_number)
        # # plotSomeInDataloader(dataloader=dataloader)
        train(dataloader=trainDataloader)
        netG, netD = loadModel(chosen_number=chosen_number)
        fakeImgs = generateFake(
            netG=netG, netD=netD, real_label=chosen_number, numImgs=800
        )
        # Save the generated imamges to folder
        for i in range(len(fakeImgs)):
            vutils.save_image(fakeImgs[i], f"fakeImgs/{chosen_number}/{i}.jpg")
