import cv2
import numpy as np
from DataPreprocess import read_imgs


def get_classifier_HOG_SVM(train_imgs, train_labels):
    """
    Get a SVM classifier with HOG feature. Use HOG from skimage.
    Use cv2.hog
    train_imgs is 64 * 64 images.
    """
    # HOG
    # First parameter is the size of the image, 
    # second is the size of the block, 
    # third is the size of the block stride, 
    # fourth is the size of the cell, 
    # fifth is the number of orientation bins.
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    train_des = []
    for img in train_imgs:
        img = np.uint8(img)
        des = hog.compute(img)
        train_des.append(des)
    train_des = np.array(train_des)
    train_des = train_des.reshape(len(train_des), -1)
    train_des = np.float32(train_des)
    # SVM
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.train(train_des, cv2.ml.ROW_SAMPLE, train_labels)
    return svm, hog

if __name__ == "__main__":
    test_imgs, test_labels, train_imgs, train_labels = read_imgs()
    img = test_imgs[120]
    # Normalize
    img = np.float32(img)/255.0
    # Calculate grad x and grad y
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # Calculate bin
    bin_n = 9
    bin = np.int32(bin_n*ang/360.0)
    # Calculate histogram
    bin_cells = []
    mag_cells = []
    cellx = celly = 8
    for i in range(0, img.shape[0], celly):
        for j in range(0, img.shape[1], cellx):
            bin_cells.append(bin[i:i+celly, j:j+cellx])
            mag_cells.append(mag[i:i+celly, j:j+cellx])
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # import matplotlib.pyplot as plt
    # # Plot bins
    # plt.bar(range(9), hist[:9])
    # plt.show()


