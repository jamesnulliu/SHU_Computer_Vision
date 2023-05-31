# 程序中的datatrain.txt和datatest.txt中，要去掉rensor，否则读取文件时会报错
import torch
from cv2 import imread
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import h5py
import time

import dataReader
import CNN_Classifier
import HandDetector

trainRoot = "trainImgs"
fakeRoot = "fakeImgs"
testRoot = "testImgs"
WhetherFake = True

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)


if WhetherFake:
    classifier = CNN_Classifier.getClassifier(
        CNN_Classifier.CLASSIFIER_ROOT_EXTEND + "/epoch9.pth"
    )
else:
    classifier = CNN_Classifier.getClassifier(
        CNN_Classifier.CLASSIFIER_ROOT_ORIGIN + "/epoch9.pth"
    )

classifier.eval()


def test():
    data = h5py.File(testRoot + "/testData.h5", "r")
    testImgs = np.array(data["X"])
    testLabels = np.array(data["Y"])
    testImgs = np.transpose(testImgs, (0, 3, 1, 2))
    # Choose the imgs whose label is 0,1,2,4,5
    testImgs = testImgs[
        np.where(
            (testLabels == 0)
            | (testLabels == 1)
            | (testLabels == 2)
            | (testLabels == 4)
            | (testLabels == 5)
        )
    ]
    testLabels = testLabels[
        np.where(
            (testLabels == 0)
            | (testLabels == 1)
            | (testLabels == 2)
            | (testLabels == 4)
            | (testLabels == 5)
        )
    ]
    # Normalize imgs
    testImgs = testImgs / 255.0
    testImgs = torch.tensor(testImgs).type(torch.float32)
    testLabels = torch.tensor(testLabels)
    test_dataset = torch.utils.data.TensorDataset(testImgs, testLabels)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=50, shuffle=True
    )
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        flipped = torch.flip(images, [3])
        total += labels.size(0)
        images = torch.cat((images, flipped), 0)
        labels = torch.cat((labels, labels), 0)
        outputs = classifier(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        # if an flipped image is predicted as label, and the original image is also predicted as label, minus 1 in correct
        originPred = (
            predicted[0 : int(labels.size(0) / 2)]
            == labels[0 : int(labels.size(0) / 2)]
        )
        flippedPred = (
            predicted[int(labels.size(0) / 2) : labels.size(0)]
            == labels[int(labels.size(0) / 2) : labels.size(0)]
        )
        correct -= (originPred & flippedPred).sum()
    print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))


def testForVedio():
    pTime=0
    cTime=0
    cap = cv2.VideoCapture(0)
    detector = HandDetector.HandDetector()
    classifier.eval()

    while True:
        _, img = cap.read()
        img = detector.findHands(img)
        lmlist, boundingBox = detector.findPosition(img=img, draw=False)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime

        if len(lmlist) > 0:
            leftmost, rightmost, topmost, bottommost = boundingBox
            w = rightmost - leftmost
            h = bottommost - topmost
            if w > h:
                topmost = topmost - (w - h) / 2
                bottommost = bottommost + (w - h) / 2
            else:
                leftmost = leftmost - (h - w) / 2
                rightmost = rightmost + (h - w) / 2

            leftmost = leftmost - 20
            rightmost = rightmost + 20
            topmost = topmost - 20
            bottommost = bottommost + 20

            handBox = img[int(topmost) : int(bottommost), int(leftmost) : int(rightmost)]

            if(handBox.shape[0] != 0 and handBox.shape[1] != 0):
                rgb = cv2.resize(handBox, (64, 64))
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = np.array(rgb, dtype=np.float32)
                rgb = rgb / 255.0
                rgb = rgb.transpose((2,0,1))
                rgb = torch.tensor(rgb).type(torch.float32)
                rgb = rgb.unsqueeze(0)
                outputs = classifier(rgb)
                pred_y1 = torch.max(outputs, 1)[1].data.numpy()
                possibility1 = torch.max(outputs, 1)[0].data.numpy()
                # Flip img in horizon
                rgb = torch.flip(rgb, [3])
                outputs = classifier(rgb)
                pred_y2 = torch.max(outputs, 1)[1].data.numpy()
                possibility2 = torch.max(outputs, 1)[0].data.numpy()
                pred_yReal = 0
                if possibility1 > possibility2:
                    pred_yReal = pred_y1
                else:
                    pred_yReal = pred_y2
                # Print the result on the img
                cv2.putText(
                    img,
                    str(pred_yReal),
                    (int(leftmost), int(topmost) - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (0, 255, 0),
                    3,
                )

            # Draw a blue bounding box
            cv2.rectangle(
                img=img,
                pt1=(int(leftmost), int(topmost)),
                pt2=(int(rightmost), int(bottommost)),
                color=(0, 255, 0),
                thickness=2,
            )

        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )

        # If q is pressed, quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.imshow("Image", img)

if __name__ == "__main__":
    # test()
    testForVedio()