import cv2
import os

os.makedirs("image/1", exist_ok=True)
os.makedirs("image/2", exist_ok=True)
os.makedirs("image/3", exist_ok=True)
os.makedirs("image/4", exist_ok=True)
os.makedirs("image/5", exist_ok=True)
os.makedirs("image/6", exist_ok=True)
os.makedirs("image/7", exist_ok=True)
os.makedirs("image/8", exist_ok=True)
os.makedirs("image/9", exist_ok=True)
os.makedirs("image/0", exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
i = 51

dataLabel =  9

while True:
    if i == 100:
        break

    ret, frame = cap.read()
    # Cut the center square of frame
    capedImg = frame[100:500, 100:500]

    cv2.imshow("frame", capedImg)
    # Wait for an input

    if (cv2.waitKey(1) & 0xFF) == ord("s"):
        capedImg = cv2.resize(capedImg, (128, 128))
        cv2.imwrite("image/" + str(dataLabel) + "/" + str(i) + ".jpg", capedImg)
        i += 1
    elif (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
