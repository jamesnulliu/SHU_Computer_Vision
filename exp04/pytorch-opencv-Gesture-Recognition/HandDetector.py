"""
Hand tracking Module
Name: Syed
"""


import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        boundingBox = (0, 0, 0, 0)
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            # landmarks are the 21 points on the hand
            leftmost = myHand.landmark[0].x
            rightmost = myHand.landmark[0].x
            topmost = myHand.landmark[0].y
            bottommost = myHand.landmark[0].y

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                leftmost = min(leftmost, lm.x)
                rightmost = max(rightmost, lm.x)
                topmost = min(topmost, lm.y)
                bottommost = max(bottommost, lm.y)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])

            boundingBox = (leftmost * w, rightmost * w, topmost * h, bottommost * h)
        return lmlist, boundingBox


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        _, img = cap.read()
        img = detector.findHands(img)
        _, boundingBox = detector.findPosition(img)
        # if len(lmlist) != 0:
        #     print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Adjust bounding box to be a square
        # # The square is centered at the center of the rectangle
        leftmost, rightmost, topmost, bottommost = boundingBox
        w = rightmost - leftmost
        h = bottommost - topmost
        if w > h:
            topmost = topmost - (w - h) / 2
            bottommost = bottommost + (w - h) / 2
        else:
            leftmost = leftmost - (h - w) / 2
            rightmost = rightmost + (h - w) / 2

        # Append 20 px to the bounding box
        leftmost = leftmost - 20
        rightmost = rightmost + 20
        topmost = topmost - 20
        bottommost = bottommost + 20

        hand_img = img[int(topmost) : int(bottommost), int(leftmost) : int(rightmost)]

        # Draw a blue bounding box
        cv2.rectangle(
            img=img,
            pt1=(int(leftmost), int(topmost)),
            pt2=(int(rightmost), int(bottommost)),
            color=(0, 255, 0),
            thickness=2,
        )

        cv2.putText(
            img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )

        cv2.imshow("Image", img)
        cv2.waitKey(1)

        


if __name__ == "__main__":
    main()
