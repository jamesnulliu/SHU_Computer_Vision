import numpy as np
import cv2 as cv
img = cv.imread(r"cv_exp01/img/tree.jpg",1)

def showImg(outputLoc, image):
    cv.namedWindow('showImg',cv.WINDOW_NORMAL)
    cv.imshow('showImg',image)
    key = cv.waitKey(0) & 0xFF
    if key == 27:           # IF pressed ESC, quit
        cv.destroyAllWindows()
    elif key == ord('s'):
        cv.imwrite(outputLoc,image)
        cv.destroyAllWindows()

def splitChannels(image):
    b,g,r = cv.split(image)
    return [b,g,r]

def showBGR():
    [b,g,r] = splitChannels(img)
    showImg(r'cv_exp01/output/blue.jpg',b)
    showImg(r'cv_exp01/output/green.jpg',g)
    showImg(r'cv_exp01/output/red.jpg',r)

def setName():
    cv.putText(img,'21121319 Liu Yanchen',(4,30), 4, 1,(255,255,255),2,cv.LINE_AA)
    showImg(r'cv_exp01/output/nameAdded.png', img)

def copyTree(destY,destX,sourceY,sourceX,yLength,xLength,resizeY,resizeX):
    tree = img[sourceY:sourceY+yLength,sourceX:sourceX+xLength]
    print(tree.shape)
    resizedtree = cv.resize(tree,(0,0), None,resizeX, resizeY,interpolation = cv.INTER_CUBIC)
    img[destY:destY+resizedtree.shape[0],destX:destX+resizedtree.shape[1]] = resizedtree 

# copyTree(190,133,190,530,315,295,1.0,0.9)
# showImg("",img)

