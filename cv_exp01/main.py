import cv2

img = cv2.imread(r"cv_exp01/img/tree.jpg",1)

res = cv2.resize(img,(0,0),None, 0.8,0.8)

cv2.namedWindow('showImg',cv2.WINDOW_NORMAL)
cv2.imshow('showImg',img)

