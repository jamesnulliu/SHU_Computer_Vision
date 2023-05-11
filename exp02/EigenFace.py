import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os

trainImg_Num = 8
testImg_Num = 2

def readTestImgsAndLabels():
    """
    This function reads test images in the att_imgs folder and returns a list of test images and lables.

    returns: a [N x (H * W)] np array (float32) of test images,
             and a [N x 1] np array (int) of labels
    """
    imgs = []
    labels = []
    for i in range(1, 41):
        for j in range(9, 11):
            img = cv2.imread('att_imgs/s' + str(i) + '/' + str(j) + '.pgm', 0)
            imgs.append(img)
            labels.append(i)
    imgs = np.array(imgs)
    imgs = imgs.reshape(imgs.shape[0],-1)
    labels = np.array(labels)
    return imgs.astype('float32'), labels

def readTrainImgsAndLabels():
    """
    This function reads train images in the att_imgs folder and returns a list of train images and lables.

    returns: a [N x (H * W)] np array (float32) of train images,
             and a [N x 1] np array (int) of labels
    """ 
    imgs = []
    labels = []
    for i in range(1, 41):
        for j in range(1, 9):
            img = cv2.imread('att_imgs/s' + str(i) + '/' + str(j) + '.pgm', 0)
            imgs.append(img)
            labels.append(i)
    imgs = np.array(imgs)
    imgs = imgs.reshape(imgs.shape[0],-1)
    labels = np.array(labels)
    return imgs.astype('float32'), labels

def saveAllImgs2Png():
    """
    This function saves all images in the att_imgs folder as .png files.
    """
    for i in range(1, 41):
        for j in range(1, 11):
            img = cv2.imread('att_imgs/s' + str(i) + '/' + str(j) + '.pgm', 0)
            cv2.imwrite('att_imgs/s' + str(i) + '/' + str(j) + '.png', img)
    
def getBaseVecs(images2D, K):
    """
    This function calculates the engine imgs of the original images.
    The orthogonalized engine imgs and average img are saved into a .npy file.

    images2D: a [N x (W * H)] np array containing the original images
    K: the number of engine imgs
    """
    # Calculate the average img
    averageimg = np.mean(images2D, axis=0)
    averageimg = averageimg.astype('float32')

    # Calculate the covariance matrix ([N x N]) of the result mat
    covMat = np.cov(images2D)  

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    # eigenVecs is a [N x N] mat
    eigenValues, eigenVecs = np.linalg.eig(covMat)
    
    # Calculate the engine imgs of original images
    # Now eigenVecs is a [N x (W * H)] mat
    eigenVecs = np.dot(eigenVecs, images2D - averageimg)
    
    # Get the indexes of the K largest eigenvalues
    indexes = np.argsort(eigenValues)[::-1][:K]

    # Get the K largest eigenvalues
    eigenValues = eigenValues[indexes].copy()
    # Get the K largest eigenvectors, i.e., the engine imgs
    eigenVecs = eigenVecs[indexes].copy()

    # Orthogonalize the engine imgs
    orth_eigenVecs, _ = np.linalg.qr(eigenVecs.T)
    orth_eigenVecs = orth_eigenVecs.T

    # Save the orthogonalized engine imgs and average img into a .npy file
    np.save('orth_engineVecs.npy', [orth_eigenVecs, averageimg])

def trainimgs(trainImgs, labels):
    """
    This function stores the train img into a .npy file.

    imgImgs: a [N x H x W] numpy array containing the train imgs
    labels: a [N x 1] numpy array containing the labels of the train imgs
    """
    orth_eigenVecs, averageimg = np.load('orth_engineVecs.npy', allow_pickle=True)

    # reshape imgImgs to (N, HW) matrix
    trainImgs = trainImgs.reshape(trainImgs.shape[0], -1)
    trainImgs = trainImgs.astype('float64')
    # reshape averageimg to (1, HW) matrix
    averageimg = averageimg.reshape(1, -1)
    # Subtract averageimg from imgImgs
    trainImgs -= averageimg
    # Calculate the engine values of the train imgs
    engineValues = np.dot(trainImgs, orth_eigenVecs.T)  # (N, K) matrix
    labels = labels.reshape(-1,1)
    data = np.concatenate((engineValues, labels), axis=1)
    # Store the train imgs, engine values and labels into a .npy file
    np.save('trainimgs.npy', data)


def predict(testimg):
    """
    This function predicts the label of a test img.

    testimg: a [H x W] numpy array containing the test img

    return: the label of the test img
    """
    # Load the train imgs, engine values and labels from the .npy file
    data = np.load('trainimgs.npy', allow_pickle=True)
    col = data.shape[1]
    engineValues = data[:, range(0, col - 1)]
    labels = data[:, col - 1]

    # Load the orthogonalized engine imgs and average img from the .npy file
    orth_engineVecs, averageimg = np.load('orth_engineVecs.npy', allow_pickle=True)

    # reshape testimg to [1 x (H * W)]
    testimg = testimg.reshape(1, -1)
    # Subtract averageimg from testimg
    testimg -= averageimg
    # Calculate the engine value of the test img
    testEngineValue = np.dot(testimg, orth_engineVecs.T)
    # Calculate the distance between the test img and the train imgs
    distances = np.linalg.norm(engineValues - testEngineValue, axis=1)
    # Get the index of the train img with the minimum distance
    index = np.argmin(distances)
    # Get the label of the train img with the minimum distance
    label = labels[index]

    return label

if __name__ == '__main__':
    train, trainLables = readTrainImgsAndLabels()
    test, testLables = readTestImgsAndLabels()

    ks = [5, 10, 15, 20,30,40,60,80,100,140,180]
    accuracies = [0.825, 0.9, 0.9375, 0.95,0.9625, 0.9625,0.9625, 0.95, 0.9625, 0.9625, 0.9625]

    getBaseVecs(train, 30)
    trainimgs(train, trainLables)
    correct = 0
    for i in range(len(test)):
        if predict(test[i]) == testLables[i]:
            correct += 1
    print(float(correct)/len(test))

    # draw plot of k vs accuracy
    # plt.plot(ks, accuracies)
    # plt.xlabel('k')
    # plt.ylabel('accuracy')
    # plt.show()