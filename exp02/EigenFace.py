import cv2
import numpy as np

trainImg_Num = 8
testImg_Num = 2
K = 3

def readTestImgsAndLabels():
    """
    This function reads test images in the att_faces folder and returns a list of test images and lables.

    returns: a [N x (H * W)] np array (float32) of test images,
             and a [N x 1] np array (int) of labels
    """
    imgs = []
    labels = []
    for i in range(1, 41):
        for j in range(9, 11):
            img = cv2.imread('att_faces/s' + str(i) + '/' + str(j) + '.pgm', 0)
            imgs.append(img)
            labels.append(i)
    imgs = np.array(imgs)
    imgs = imgs.reshape(imgs.shape[0],-1)
    labels = np.array(labels)
    return imgs.astype('float32'), labels

def readTrainImgsAndLabels():
    """
    This function reads train images in the att_faces folder and returns a list of train images and lables.

    returns: a [N x (H * W)] np array (float32) of train images,
             and a [N x 1] np array (int) of labels
    """ 
    imgs = []
    labels = []
    for i in range(1, 41):
        for j in range(1, 9):
            img = cv2.imread('att_faces/s' + str(i) + '/' + str(j) + '.pgm', 0)
            imgs.append(img)
            labels.append(i)
    imgs = np.array(imgs)
    imgs = imgs.reshape(imgs.shape[0],-1)
    labels = np.array(labels)
    return imgs.astype('float32'), labels

def saveAllImgs2Png():
    """
    This function saves all images in the att_faces folder as .png files.
    """
    for i in range(1, 41):
        for j in range(1, 11):
            img = cv2.imread('att_faces/s' + str(i) + '/' + str(j) + '.pgm', 0)
            cv2.imwrite('att_faces/s' + str(i) + '/' + str(j) + '.png', img)
    
def getEngineFaces(images2D, K):
    """
    This function calculates the engine faces of the original images.
    The orthogonalized engine faces and average face are saved into a .npy file.

    images2D: a [N x (W * H)] np array containing the original images
    K: the number of engine faces
    """

    # Calculate the average face
    averageFace = np.mean(images2D, axis=0)
    averageFace = averageFace.astype('float32')

    # Calculate the covariance matrix ([N x N]) of the result mat
    covMat = np.cov(images2D)  

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    # eigenVecs is a [N x N] mat
    eigenValues, eigenVecs = np.linalg.eig(covMat)
    
    # Calculate the engine faces of original images
    # Now eigenVecs is a [N x (W * H)] mat
    eigenVecs = np.dot(eigenVecs, images2D - averageFace)
    
    # Get the indexes of the K largest eigenvalues
    indexes = np.argsort(eigenValues)[::-1][:K]

    # Get the K largest eigenvalues
    eigenValues = eigenValues[indexes].copy()
    # Get the K largest eigenvectors, i.e., the engine faces
    eigenVecs = eigenVecs[indexes].copy()

    # Orthogonalize the engine faces
    orth_eigenVecs, _ = np.linalg.qr(eigenVecs.T)
    orth_eigenVecs = orth_eigenVecs.T

    # Save the orthogonalized engine faces and average face into a .npy file
    np.save('orth_engineVecs.npy', [orth_eigenVecs, averageFace])

def trainFaces(trainImgs, labels):
    """
    This function stores the train face into a .npy file.

    faceImgs: a [N x H x W] numpy array containing the train faces
    labels: a [N x 1] numpy array containing the labels of the train faces
    """
    orth_eigenVecs, averageFace = np.load('orth_engineVecs.npy', allow_pickle=True)

    # reshape faceImgs to [N x (H * W)]
    trainImgs = trainImgs.reshape(trainImgs.shape[0], -1)
    trainImgs = trainImgs.astype('float64')
    # reshape averageFace to [1 x (H * W)]
    averageFace = averageFace.reshape(1, -1)
    # Subtract averageFace from faceImgs
    trainImgs -= averageFace
    # Calculate the engine values of the train faces
    engineValues = np.dot(trainImgs, orth_eigenVecs.T)  # [N x K]
    labels = labels.reshape(-1,1)
    data = np.concatenate((engineValues, labels), axis=1)
    # Store the train faces, engine values and labels into a .npy file
    np.save('trainFaces.npy', data)


def predict(testFace):
    """
    This function predicts the label of a test face.

    testFace: a [H x W] numpy array containing the test face

    return: the label of the test face
    """
    # Load the train faces, engine values and labels from the .npy file
    data = np.load('trainFaces.npy', allow_pickle=True)
    engineValues = data[:,range(0,K)]
    labels = data[:,K]
    # engineValues, labels = np.load('trainFaces.npy', allow_pickle=True)

    # Load the orthogonalized engine faces and average face from the .npy file
    orth_engineVecs, averageFace = np.load('orth_engineVecs.npy', allow_pickle=True)

    # reshape testFace to [1 x (H * W)]
    testFace = testFace.reshape(1, -1)
    # Subtract averageFace from testFace
    testFace -= averageFace
    # Calculate the engine value of the test face
    testEngineValue = np.dot(testFace, orth_engineVecs.T)
    # Calculate the distance between the test face and the train faces
    distances = np.linalg.norm(engineValues - testEngineValue, axis=1)
    # Get the index of the train face with the minimum distance
    index = np.argmin(distances)
    # Get the label of the train face with the minimum distance
    label = labels[index]

    return label

if __name__ == '__main__':
    train, trainLables = readTrainImgsAndLabels()
    test, testLables = readTestImgsAndLabels()
    getEngineFaces(train, K)
    trainFaces(train, trainLables)
    correct = 0
    for i in range(len(test)):
        if predict(test[i]) == testLables[i]:
            correct += 1
    print('Accuracy: ' + str(correct / len(test)))

