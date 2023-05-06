import cv2
import numpy as np
import os

def read_imgs():
    """
    Read images from number_data folder and return a list of test_imgs, a list of test_labels, a list of train_imgs and a list of train_labels.
    Type of test_imgs and train_imgs is np.array of uint8, and the shape of each element is (30, 30).
    """
    test_imgs = []
    test_labels = []
    train_imgs = []
    train_labels = []
    for folder in ["testing", "training"]:
        path = os.path.join("number_data", folder)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (60, 60))
            img = np.array(img, dtype=np.int8)
            label = int(img_name[0])
            if folder == "testing":
                test_imgs.append(img)
                test_labels.append(label)
            else:
                train_imgs.append(img)
                train_labels.append(label)
    test_imgs = np.array(test_imgs)
    test_labels = np.array(test_labels)
    train_imgs = np.array(train_imgs)
    train_labels = np.array(train_labels)
    return test_imgs, test_labels, train_imgs, train_labels

def increaseData(imgs, labels):
    """
    Increase the number of data by flipping the original data.

    Returns A list of imgs and a list of labels.
    """
    new_imgs = []
    new_labels = []
    for i in range(len(imgs)):
        img = imgs[i]
        label = labels[i]
        img = img.reshape(60, 60)
        new_imgs.append(img)
        new_labels.append(label)
        new_imgs.append(cv2.flip(img, 0))
        new_labels.append(label)
        new_imgs.append(cv2.flip(img, 1))
        new_labels.append(label)
    new_imgs = np.array(new_imgs)
    new_labels = np.array(new_labels)
    return new_imgs, new_labels
    
if __name__ == "__main__":
    test_imgs, test_labels, train_imgs, train_labels = read_imgs()
    print(test_imgs.shape)
    print(test_labels.shape)
    print(train_imgs.shape)
    print(train_labels.shape)