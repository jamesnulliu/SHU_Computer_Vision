import cv2

# read images
origin_img1 = cv2.imread('img/1-a.jpg')
origin_img1 = img1 = cv2.resize(origin_img1, (0, 0), fx=0.5, fy=0.5)
origin_img2 = cv2.imread('img/1-b.jpg')
origin_img2 = img2 = cv2.resize(origin_img2, (0, 0), fx=0.5, fy=0.5)

# dectect faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces1 = face_cascade.detectMultiScale(img1, 1.02, 5)
faces2 = face_cascade.detectMultiScale(img2, 1.02, 5)

# show result
for (x, y, w, h) in faces1:
    cv2.rectangle(origin_img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
for (x, y, w, h) in faces2:
    cv2.rectangle(origin_img2, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('img1', origin_img1)
cv2.imshow('img2', origin_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()