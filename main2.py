from Modules.grabscreen import set_screen
from Modules.directkeys import PressKey, ReleaseKey, W, A, D, S
from Modules.getkeys import key_check
from Modules.alexnet import alexnet
from cvzone.HandTrackingModule import HandDetector
import time
import cv2

#self
WIDTH = 256
HEIGHT = 160
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'Models/euro-truck-fast-{}-{}-{}-epochs-300K-data.model'.format(
    LR, 'alexnetv2', EPOCHS)

t_time = 0.1

#motion
detector = HandDetector(maxHands=2)

cap_cam = cv2.VideoCapture(0)

class Hand:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(W)


def left():
    PressKey(W)
    PressKey(A)
    # ReleaseKey(A)
    # ReleaseKey(W)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(W)
    ReleaseKey(A)


def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    # ReleaseKey(W)
    # ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(W)
    ReleaseKey(D)

def back():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(W)
    time.sleep(t_time)
    ReleaseKey(S)


model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)


last_time = time.time()
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

paused = False
while(True):
    if not paused:
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        screen = set_screen()

        prediction = model.predict([screen.reshape(256, 160, 1)])[0]
        print(prediction)

        turn_thresh = .75
        fwd_thresh = 0.70

        if prediction[1] > fwd_thresh:
            straight()
        elif prediction[0] > turn_thresh:
            left()
        elif prediction[2] > turn_thresh:
            right()
        else:
            straight()

    keys = key_check()

    # pause
    if 'T' in keys:
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            time.sleep(1)

    else:    
        ret, img = cap_cam.read()
        if not ret:
            break

        img = cv2.flip(img, 1)

        hands, img = detector.findHands(img)

        if len(hands) == 2:
            hand1 = hands[0]
            hand2 = hands[1]
            lm_list1 = hand1['lmList']
            lm_list2 = hand2['lmList']

            # length, info = detector.findDistance(lm_list1[8], lm_list2[8])
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
            paused=True
