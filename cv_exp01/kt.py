import cv2
import pic_manipulate as e01
from pic_manipulate import img

# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 用 Sobel 算子计算梯度
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# 模糊图像2值化
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

closed = cv2.erode(thresh, None, iterations=4)
 
closed = cv2.dilate(closed, None, iterations=4)

e01.showImg(r'cv_exp01/output/nameAdded.png', closed)
