import cv2
import numpy as np
from DataPreprocess import read_imgs
from PCA_KNN import get_classifier_PCA_KNN
from HOG_SVM import get_classifier_HOG_SVM
from CNN import get_classifier_CNN
from CNN import process_data
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

def test_PCA_KNN(K):
    test_imgs, test_labels, train_imgs, train_labels = read_imgs()
    
    train_time_start = cv2.getTickCount()
    knn, pca = get_classifier_PCA_KNN(train_imgs, train_labels, K)
    train_time_stop = cv2.getTickCount()
    train_time = (train_time_stop - train_time_start)/cv2.getTickFrequency()

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
    accuracy = correct/float(len(test_imgs))
    return accuracy, train_time

def test_HOG_SVM():
    test_imgs, test_labels, train_imgs, train_labels = read_imgs()

    train_time_start = cv2.getTickCount()
    svm, hog = get_classifier_HOG_SVM(train_imgs, train_labels)
    train_time_stop = cv2.getTickCount()

    print("SVM Training time: ", (train_time_stop - train_time_start)/cv2.getTickFrequency())
    # Test with classifier of HOG + SVM
    print("Testing with classifier of HOG + SVM...", end=" ")
    correct = 0
    for i in range(len(test_imgs)):
        img = test_imgs[i]
        img = np.uint8(img)
        label = test_labels[i]
        # HOG
        des = hog.compute(img)
        # SVM
        des = des.reshape(1, -1)
        des = des.astype(np.float32)
        _, result = svm.predict(des)
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
    # PCAKNN_Ks = [40, 45, 50, 55,  60, 65, 70, 80, 90, 100, 110, 120, 130, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    # # PCAKNN_Ks = [40, 45, 50]
    # accuracies = []
    # train_times = []
    # for k in PCAKNN_Ks:
    #     print("PCA + KNN, K = ", k)
    #     accuracy, train_time = test_PCA_KNN(k)
    #     accuracies.append(accuracy)
    #     train_times.append(train_time)
    # # Plot both accuracy and tainning time in same figure under different Ks
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(PCAKNN_Ks, accuracies)
    # plt.ylabel("Accuracy")
    # plt.title("PCA + KNN")
    # plt.subplot(212)
    # plt.plot(PCAKNN_Ks, train_times)
    # plt.xlabel("K")
    # plt.ylabel("Training time")
    # plt.show()
    test_HOG_SVM()
    