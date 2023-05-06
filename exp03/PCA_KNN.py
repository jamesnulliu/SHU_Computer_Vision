import cv2
import numpy as np
from sklearn.decomposition import PCA

def get_classifier_PCA_KNN(train_imgs, train_labels):
    """
    Get a KNN classifier with PCA feature. Use PCA class from sklearn.decomposition.
    """
    # PCA
    pca = PCA(n_components=0.9)
    # Flatern train_imgs
    train_imgs = train_imgs.reshape(len(train_imgs), -1)
    # Cast train_imgs to F32
    train_imgs = train_imgs.astype(np.float32)
    pca.fit(train_imgs)
    train_des = pca.transform(train_imgs)
    # KNN
    knn = cv2.ml.KNearest_create()
    knn.train(train_des, cv2.ml.ROW_SAMPLE, train_labels)
    return knn, pca