import cv2
import pic_manipulate as e01


def sobel(img):
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 用 Sobel 算子计算梯度
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    e01.showImg(gradient, "gradient")
    # 图像2值化
    (_, thresh) = cv2.threshold(gradient, 90, 255, cv2.THRESH_BINARY)
    # 形态学腐蚀与膨胀
    thresh = cv2.erode(thresh, None, iterations = 2)
    thresh = cv2.dilate(thresh, None, iterations = 5)
    return thresh
