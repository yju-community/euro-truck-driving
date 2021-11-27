from Modules.grabscreen import set_screen
from Modules.directkeys import PressKey, ReleaseKey, W, A, D
from Modules.getkeys import key_check
from Modules.alexnet import alexnet
import time

WIDTH = 256
HEIGHT = 160
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'Models/euro-truck-fast-{}-{}-{}-epochs-300K-data.model'.format(
    LR, 'alexnetv2', EPOCHS)

t_time = 0.1


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
