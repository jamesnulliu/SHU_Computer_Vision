import cv2
import numpy as np
from sklearn.decomposition import PCA

def get_classifier_PCA_KNN(train_imgs, train_labels, K = 40):
    """
    Get a KNN classifier with PCA feature. Use PCA class from sklearn.decomposition.
    """

    # PCA
    pca = PCA(n_components=K)
    # Flatern train_imgs
    train_imgs = train_imgs.reshape(len(train_imgs), -1)
    # Cast train_imgs to F32
    train_imgs = train_imgs.astype(np.float32)
    pca.fit(train_imgs)
    train_des = pca.transform(train_imgs)

    # [Optional] Show feature vectors
    # featureVectors = pca.components_
    # for i in range(40):
    #     featureVector = featureVectors[i]
    #     featureVector = featureVector.reshape(60, 60)
    #     featureVector = cv2.resize(featureVector, (300, 300))
    #     featureVector = (featureVector - np.min(featureVector))/(np.max(featureVector) - np.min(featureVector))
    #     cv2.imshow("featureVector", featureVector)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # KNN
    knn = cv2.ml.KNearest_create()
    knn.train(train_des, cv2.ml.ROW_SAMPLE, train_labels)
    return knn, pca