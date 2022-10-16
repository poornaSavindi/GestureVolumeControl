# we are gonna modularize the previous code.
# so we can use it easily on another project.

import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, maxhands=2, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.maxHands = maxhands
        self.detectionCon = detectioncon
        self.trackCon = trackcon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)  # gets the hands set in mediapipe module

        self.mpDraw = mp.solutions.drawing_utils
        # a class tht has methos to draw landmarks on the hand

    def findHands(self, img, draw=True):
        # now we have to send the rgb image to the hands object
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.result = self.hands.process(imgRGB)
        # Processes an RGB image and returns the hand landmarks and handedness of each detected hand.

        if draw:
            if self.result.multi_hand_landmarks:
                for handlms in self.result.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
                    # draw the lines connecting 21 different landmarks

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.result.multi_hand_landmarks:
            for id, lm in enumerate(self.result.multi_hand_landmarks[handNo].landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # now we have got the pixel values of x,y cordinates of landmarks
                lmList.append([id, cx, cy])

        return lmList
