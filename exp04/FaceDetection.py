import cv2

# read imgs in 'img' folder in grayscale
img1 = cv2.imread('img/1-a.jpg', 0)
img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
img2 = cv2.imread('img/1-b.jpg', 0)
img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)

# dectect faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces1 = face_cascade.detectMultiScale(img1, 1.05, 3)
faces2 = face_cascade.detectMultiScale(img2, 1.05, 5)

# show result
for (x, y, w, h) in faces1:
    cv2.rectangle(img1, (x, y), (x+w, y+h), (255, 0, 0), 2)
for (x, y, w, h) in faces2:
    cv2.rectangle(img2, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()