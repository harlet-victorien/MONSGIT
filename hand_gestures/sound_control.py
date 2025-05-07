import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import math
import comtypes



# Initialiser la détection des mains
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialiser le volume système
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Récupérer la plage de volume
volMin, volMax = volume.GetVolumeRange()[:2]

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                lmList.append((int(lm.x * w), int(lm.y * h)))
            
            # Distance entre pouce (id=4) et index (id=8)
            x1, y1 = lmList[4]
            x2, y2 = lmList[8]
            cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
            distance = math.hypot(x2 - x1, y2 - y1)

            # Mapper la distance à un niveau de volume
            vol = np.interp(distance, [20, 120], [volMin, volMax])
            volume.SetMasterVolumeLevel(vol, None)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break