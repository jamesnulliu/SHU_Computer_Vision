import cv2
import numpy as np
from skimage.feature import hog


def get_classifier_HOG_SVM(train_imgs, train_labels):
    """
    Get a SVM classifier with HOG feature. Use HOG from skimage.
    """
    # HOG
    train_des = []
    for img in train_imgs:
        des = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        train_des.append(des)
    train_des = np.array(train_des)
    # SVM
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # Cast train_des to 32F
    train_des = train_des.astype(np.float32)
    svm.train(train_des, cv2.ml.ROW_SAMPLE, train_labels)
    return svm
