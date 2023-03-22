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

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'21121319 Liu Yanchen',(4,30), font, 1,(255,255,255),2,cv.LINE_AA)
showImg(r'cv_exp01/output/nameAdded.png', img)

tree = img[514:831, 183:504]
img[0:(831-514), 0:(504-183)] = tree
showImg(r'cv_exp01/output/nameAdded.png', img)