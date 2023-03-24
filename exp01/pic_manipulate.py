import cv2
from data import originImage

def showImg(image, windowName):
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.imshow(windowName, image)

def wait4close(image = 0, outputLoc = ""):
    key = cv2.waitKey(0) & 0xFF
    while True:
        if key == 27:  # IF pressed ESC, quit
            cv2.destroyAllWindows()
            break
        key = cv2.waitKey(0) & 0xFF

def showBGR(originImg):
    [b, g, r] = cv2.split(originImg)
    cv2.imwrite(r'output/blue.png', b)
    cv2.imwrite(r'output/green.png', g)
    cv2.imwrite(r'output/red.png', r)
    showImg(b, 'blue')
    showImg(g, 'green')
    showImg(r, 'red')
    wait4close()


def setName(originImg):
    newImg = originImg.copy()
    cv2.putText(newImg, '21121319 Liu Yanchen', (4, 30), 4, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return newImg


def copyTree(originImg, destY, destX, sourceY, sourceX, yLength, xLength, yTimes, xTimes):
    newImg=originImg.copy()
    tree = newImg[sourceY:sourceY + yLength, sourceX:sourceX + xLength]
    resizedTree = cv2.resize(tree, (0, 0), None, xTimes, yTimes, interpolation=cv2.INTER_CUBIC)
    newImg[destY:destY + resizedTree.shape[0], destX:destX + resizedTree.shape[1]] = resizedTree
    return newImg


def copyTree_threshold(originImg, threshold, destY, destX, sourceY, sourceX, yLength, xLength, resizeY, resizeX):
    newImg = originImg.copy()
    tree = newImg[sourceY:sourceY + yLength, sourceX:sourceX + xLength]  # 截取树的矩阵
    treeThreshold = threshold[sourceY:sourceY + yLength, sourceX:sourceX + xLength]  # 截取树对应二值化矩阵
    resizedTree = cv2.resize(tree, (0, 0), None, resizeX, resizeY, interpolation=cv2.INTER_CUBIC)  # 缩放树矩阵
    resizedThreshold = cv2.resize(treeThreshold,(0,0),None,resizeX, resizeY, interpolation=cv2.INTER_CUBIC)  #缩放对应二值化矩阵
    # 遍历每个像素, 当二值化像素点为 255 时(表示像素不属于天空), 将像素复制到相应位置
    for y in range(destY, destY + resizedTree.shape[0]):
        for x in range(destX, destX + resizedTree.shape[1]):
            if resizedThreshold[y-destY][x-destX] == 255:
                newImg[y][x] = resizedTree[y-destY][x-destX]
    return newImg

if __name__ == '__main__':
    showBGR(originImage)