import cv2
import os
import torch
import torchvision.transforms as T
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

TRAIN_ROOT = "trainImgs"

def getImagesAndLabels(root):
    imagePaths = [os.path.join(root, str(i)) for i in [0, 1, 2, 4, 5]]
    images = []
    labels = []
    for imagePath in imagePaths:
        for image in os.listdir(imagePath):
            image = cv2.imread(os.path.join(imagePath, image))
            image = cv2.resize(image, (64, 64))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            labels.append(int(imagePath[-1]))
    return images, labels

def generateImgs(srcRoot, dstRoot, times):
    transform = T.Compose([
        T.ToPILImage(),
        # Rotate -30 - 30 degrees, fill with white
        T.RandomRotation(30, fill=(161, 159, 160)),
        T.ColorJitter(brightness=0.2, hue=0., contrast=0.2, saturation=0.2),
        T.ToTensor()
    ])
    images, labels = getImagesAndLabels(srcRoot)
    for i in range(times*len(images)):
        img = transform(images[int(i/times)])
        img = img * 255
        img = img.to(torch.uint8)
        img = img.permute(1, 2, 0)
        img = img.numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dstRoot + "/" + str(labels[int(i/times)]) + "/" + "TorchTransform_" + str(i) + ".jpg", img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

