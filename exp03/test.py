import cv2
from DataPreprocess import read_imgs
from PCA_KNN import get_classifier_PCA_KNN
from HOG_SVM import get_classifier_HOG_SVM

if __name__ == "__main__":
    test_imgs, test_labels, train_imgs, train_labels = read_imgs()
    mean, eigenvectors, knn = get_classifier_PCA_KNN(test_imgs, test_labels)
    # Test with classifier of PCA + KNN
    print("Testing with classifier of PCA + KNN...", end=" ")
    correct = 0
    for i in range(len(test_imgs)):
        img = test_imgs[i]
        label = test_labels[i]
        img = img.reshape(1, -1)
        # img = cv2.PCAProject(img, mean, eigenvectors)
        _, result, _, _ = knn.findNearest(img, 1)
        if result == label:
            correct += 1
    print(correct/float(len(test_imgs)))
    print("Done.")