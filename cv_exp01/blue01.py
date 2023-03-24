import cv2


def get_threshold_sky(img, min):
    # img = cv2.blur(img, (40, 40))
    (_, threshold) = cv2.threshold(img, min, 255, cv2.THRESH_BINARY)
    # e01.showImg("",threshold)
    return threshold
