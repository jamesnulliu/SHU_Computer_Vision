import cv2
import numpy as np

def get_classifier_HOG_SVM(train_imgs, train_labels):
    """
    Get the classifier of HOG + SVM.
    """
    # HOG
    hog = cv2.HOGDescriptor()
    hog_descriptors = []
    for img in train_imgs:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)
    # SVM
    svm = cv2.ml.SVM_create()
    svm.train(train_imgs, cv2.ml.ROW_SAMPLE, train_labels)
    return hog, svm