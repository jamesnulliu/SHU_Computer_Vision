import numpy as np
import cv2 as cv
img = cv.imread(r"cv_exp01/img/tree.jpg",0)

def showImg_fixed():
    cv.imshow('image_gray_fixed', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def showImg_unfixed():
    cv.namedWindow('image_fixed',cv.WINDOW_NORMAL)
    cv.imshow('image_fixed',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def showImg():
    cv.namedWindow('showImg',cv.WINDOW_NORMAL)
    cv.imshow('showImg',img)
    key = cv.waitKey(0) & 0xFF
    if key == 27:           # IF pressed ESC, quit
        cv.destroyAllWindows()
    elif key == ord('s'):
        cv.imwrite(r'cv_exp01/output/messigray.png',img)
        cv.destroyAllWindows()

showImg()