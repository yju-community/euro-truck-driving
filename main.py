from self_driving.testModel import self_driving
from motion_driving.motion import motion_driving
from Modules.getkeys import key_check
import cv2

cap_cam = cv2.VideoCapture(0)

SET_SWITCH = False
    
while(True):
    if SET_SWITCH:
        self_driving()
    else:
        motion_driving(cap_cam)
        
    keys = key_check()
    
    if 'T' in keys:
        SET_SWITCH = not SET_SWITCH
    elif 'Q' in keys:
        break