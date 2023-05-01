import cv2
import numpy as np

def get_classifier_PCA_KNN(test_imgs, test_labels):
    """
    Get the classifier of PCA + KNN.
    """
    # PCA
    mean, eigenvectors = cv2.PCACompute(test_imgs, mean=None, maxComponents=1)
    # KNN
    knn = cv2.ml.KNearest_create()
    # Use eignvectors to train knn
    eigenvectors = np.array(eigenvectors, dtype=np.float32)
    knn.train(test_imgs, cv2.ml.ROW_SAMPLE, test_labels)
    return mean, eigenvectors, knn
