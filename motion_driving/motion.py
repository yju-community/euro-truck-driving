import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from Modules.directkeys import PressKey, ReleaseKey, W, A, D, S
from Modules.getkeys import key_check

t_time = 0.09
detector = HandDetector(maxHands=2)

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    PressKey(A)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(A)


def right():
    PressKey(D)
    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(D)


def back():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(W)
    time.sleep(t_time)
    ReleaseKey(S)


class Hand:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def motion_driving(cap_cam):    
    ret, img = cap_cam.read()

    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    if len(hands) == 2:
        hand1 = hands[0]
        hand2 = hands[1]
        lm_list1 = hand1['lmList']
        lm_list2 = hand2['lmList']

        length, info, img = detector.findDistance(
            lm_list1[8], lm_list2[8], img)  # with draw
        if hand1["type"] == 'Left':
            left_hand = Hand(info[0], info[1])
            right_hand = Hand(info[2], info[3])
        elif hand1["type"] == 'Right':
            left_hand = Hand(info[2], info[3])
            right_hand = Hand(info[0], info[1])
        cv2.putText(img, text="left %.2f %.2f" % (left_hand.x, left_hand.y), org=(
            10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0))
        cv2.putText(img, text='right %.2f %.2f' % (right_hand.x, right_hand.y), org=(
            10, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0))

        # 오른손이 left
        # 왼손이 right
        if length < 100:
            back()
        elif abs(left_hand.y - right_hand.y) < 100:
            straight()
        elif right_hand.y < left_hand.y - 100:
            right()
        elif left_hand.y < right_hand.y - 100:
            left()

    cv2.imshow('cam', img)

    if cv2.waitKey(1) == ord('t'):
        print('on self driving mode')
