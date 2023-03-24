## 二值化
import cv2
import exp01 as e01

img = cv2.imread(r"cv_exp01/output/blue.png", 0)
img = cv2.blur(img, (9, 9))
(_, img) = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)

# e01.showImg("",img)