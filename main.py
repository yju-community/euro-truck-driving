from self_driving.testModel import self_driving
from motion_driving.motion import motion_driving
from Modules.getkeys import key_check
import cv2

cap_cam = cv2.VideoCapture(0)

SET_SWITCH = False
# while(True):
#     if SET_SWITCH:
#         self_driving()
        # keys = key_check()  
        # if 'T' in keys:
        #     SET_SWITCH = not SET_SWITCH
    # else:
    #     motion_driving(cap_cam,SET_SWITCH)
    # if cv2.waitKey(1) == ord('t'):
    #     SET_SWITCH = not SET_SWITCH
    #     print(SET_SWITCH)
    # elif cv2.waitKey(1) == ord('q'):
    #     break
    
while(True):
    if SET_SWITCH:
        print('self driving')
        self_driving()
    else:
        motion_driving(cap_cam)
        print('motion driving')
    if cv2.waitKey(1) == ord('t'):
        SET_SWITCH = not SET_SWITCH
        print(SET_SWITCH)
    elif cv2.waitKey(1) == ord('q'):
        break
        

