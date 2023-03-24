import cv2 as cv


def showImg(outputLoc, image):
    cv.namedWindow('showImg', cv.WINDOW_NORMAL)
    cv.imshow('showImg', image)
    key = cv.waitKey(0) & 0xFF
    if key == 27:  # IF pressed ESC, quit
        cv.destroyAllWindows()
    elif key == ord('s'):
        cv.imwrite(outputLoc, image)
        cv.destroyAllWindows()


def splitChannels(image):
    b, g, r = cv.split(image)
    return [b, g, r]


def showBGR(image):
    [b, g, r] = splitChannels(image)
    showImg(r'cv_exp01/output/blue.jpg', b)
    showImg(r'cv_exp01/output/green.jpg', g)
    showImg(r'cv_exp01/output/red.jpg', r)


def setName(img):
    cv.putText(img, '21121319 Liu Yanchen', (4, 30), 4, 1, (255, 255, 255), 2, cv.LINE_AA)
    showImg(r'cv_exp01/output/nameAdded.png', img)


def copyTree(img, destY, destX, sourceY, sourceX, yLength, xLength, resizeY, resizeX):
    tree = img[sourceY:sourceY + yLength, sourceX:sourceX + xLength]
    print(tree.shape)
    resizedTree = cv.resize(tree, (0, 0), None, resizeX, resizeY, interpolation=cv.INTER_CUBIC)
    img[destY:destY + resizedTree.shape[0], destX:destX + resizedTree.shape[1]] = resizedTree


def copyTree_threshold(img, threshold, destY, destX, sourceY, sourceX, yLength, xLength, resizeY, resizeX):
    tree = img[sourceY:sourceY + yLength, sourceX:sourceX + xLength]
    resizedTree = cv.resize(tree, (0, 0), None, resizeX, resizeY, interpolation=cv.INTER_CUBIC)
    cv.namedWindow('origin', cv.WINDOW_NORMAL)
    cv.imshow('origin', img)
    for y in range(destY, destY + resizedTree.shape[0]):
        for x in range(destX, destX + resizedTree.shape[1]):
            if threshold[y][x] == 0:
                img[y][x] = resizedTree[y - destY][x - destX]
    cv.namedWindow('changed', cv.WINDOW_NORMAL)
    cv.imshow('changed', img)
    key = cv.waitKey(0) & 0xFF
# copyTree(190,133,190,530,315,295,1.0,0.9)
# showImg("",img)
# showBGR()
