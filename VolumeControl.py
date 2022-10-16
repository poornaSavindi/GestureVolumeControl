import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
cTime = 0
pTime = 0

# initiate pycaw for volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

while True:
    success, img = cap.read()

    # use our module get the landmarks of the hand in the img
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    # empty bar for volume
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)

    if len(lmList) != 0:
        # take the positions of index and thumb
        x_index, y_index = lmList[4][1], lmList[4][2]
        x_thumb, y_thumb = lmList[8][1], lmList[8][2]

        # take the length between two points
        cur_diff = math.sqrt(math.pow(x_index - x_thumb , 2) + math.pow((y_index - y_thumb),2))

        cv2.circle(img, (x_index, y_index), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x_thumb, y_thumb), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x_index, y_index), (x_thumb, y_thumb), (255, 0, 255), 2)

        # using interpolation, take the volume which corresponds to the length between two fingers
        cur_volume = np.interp(cur_diff,[30,200],[-65,0])
        volume.SetMasterVolumeLevel(cur_volume, None)

        cur_vol_bar = np.interp(cur_diff,[30,200],[400,150])
        cur_vol = np.interp(cur_diff, [30, 200], [0,100])


        cv2.rectangle(img, (50, int(cur_vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img,f'{str(int(cur_vol))}%',(50,130),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,255,0),2)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,f'Fps:{str(int(fps))}',(20,40),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0),2)
    cv2.imshow('WebCam', img)
    cv2.waitKey(1)
