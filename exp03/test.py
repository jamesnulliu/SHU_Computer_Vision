import cv2
import numpy as np
from DataPreprocess import read_imgs
from skimage.feature import hog
from PCA_KNN import get_classifier_PCA_KNN
from HOG_SVM import get_classifier_HOG_SVM
from CNN import get_classifier_CNN
from CNN import process_data
import torch
from torch.autograd import Variable

def test_KNN():
    test_imgs, test_labels, train_imgs, train_labels = read_imgs()
    
    train_time_start = cv2.getTickCount()
    knn, pca = get_classifier_PCA_KNN(train_imgs, train_labels)
    train_time_stop = cv2.getTickCount()
    print("KNN Training time: ", (train_time_stop - train_time_start)/cv2.getTickFrequency())


    # Test with classifier of PCA + KNN
    print("Testing with classifier of PCA + KNN...", end=" ")
    correct = 0
    for i in range(len(test_imgs)):
        img = test_imgs[i]
        label = test_labels[i]
        # Flatern img
        img = img.reshape(1, -1)
        # Cast img to F32
        img = img.astype(np.float32)
        # PCA
        des = pca.transform(img)
        # KNN
        _, result, _, _ = knn.findNearest(des, 1)
        if result[0][0] == label:
            correct += 1
    print(correct/float(len(test_imgs)))
    print("Done.")

def test_SVM():
    test_imgs, test_labels, train_imgs, train_labels = read_imgs()

    train_time_start = cv2.getTickCount()
    svm = get_classifier_HOG_SVM(train_imgs, train_labels)
    train_time_stop = cv2.getTickCount()

    print("SVM Training time: ", (train_time_stop - train_time_start)/cv2.getTickFrequency())
    # Test with classifier of HOG + SVM
    print("Testing with classifier of HOG + SVM...", end=" ")
    correct = 0
    for i in range(len(test_imgs)):
        img = test_imgs[i]
        label = test_labels[i]
        # HOG
        des = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        # SVM
        des = des.reshape(1, -1)
        des = des.astype(np.float32)
        ret, result = svm.predict(des)
        if result[0][0] == label:
            correct += 1
    print(correct/float(len(test_imgs)))
    print("Done.")

def test_CNN():
    test_imgs, test_labels, train_imgs, train_labels = read_imgs()

    train_time_start = cv2.getTickCount()
    cnn = get_classifier_CNN(train_imgs, train_labels)
    train_time_stop = cv2.getTickCount()
    print("CNN Training time: ", (train_time_stop - train_time_start)/cv2.getTickFrequency())

    # Test with classifier of CNN
    print("Testing with classifier of CNN...", end=" ")
    correct = 0
    for i in range(len(test_imgs)):
        img = test_imgs[i]
        label = test_labels[i]
        img = process_data(img)
        # CNN
        output = cnn(img)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] == label:
            correct += 1
    print(correct/float(len(test_imgs)))
    print("Done.")


if __name__ == "__main__":
    test_KNN()
    test_SVM()
    test_CNN()