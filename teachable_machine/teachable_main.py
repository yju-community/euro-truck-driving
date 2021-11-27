from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
from Modules.directkeys import PressKey, ReleaseKey, W, A, D, S
from Modules.getkeys import key_check

t_time = 0.09


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


model = load_model('teachable_machine/keras_model.h5')

cap = cv2.VideoCapture(0)

classes = ['None', 'Straight', 'Back', 'Left', 'Right']

'''
0 None
1 Straight
2 Back
3 Left
4 Right

캡쳐를 하는데 프레임 단위로 while문을 돈다
cap.isOpened 열려있는 동안
'''
i = 0
while cap.isOpened():
    ret, img = cap.read()  # ret 성공적인지 boolean / img = 이미지를 numpy array 형태로 반환

    if not ret:  # 이미지를 읽는 게 실패면 빠져나온다
        break

    # 캠이 기본적으로 좌우반전
    # 그것을 다시 좌우반전 시켜서 원래대로
    img = cv2.flip(img, 1)

    # 세로 가로 채널q
    h, w, c = img.shape

    img = img[:, 60:60+h]  # 웹캠 정사각형으로 자르기

    img_input = cv2.resize(img, (224, 224))  # 224 224형태로 리사이징

    # BGR -> RGB 바꾸는 이유는 티쳐블 머신이 RGB형태로 와서 cv에 default bgr을 rgb로 바꾸는 것
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

    img_input = (img_input.astype(np.float32) / 127.0) - 1.0
    img_input = np.expand_dims(img_input, axis=0)

    # 사람이 알아보기 힘든 값으로 나옴 모델에 이미지 학습에 대한 확률이
    # [[0.56628793 0.00919314 0.31540975 0.03834479 0.0707643 ]]
    prediction = model.predict(img_input)

    # prediction을 idx로 바꿔준다. idx 는 labels.txt에 적혀있는 idx / 학습한 모션에 대한 idx
    idx = np.argmax(prediction)

    # 출력하기
    cv2.putText(img, text=classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2, color=(0, 0, 0), thickness=1)

    # if i > 5:
    if classes[idx] == 'Straight':
        straight()
        # print('straight')
    elif classes[idx] == 'Back':
        back()
        # print('back')
    elif classes[idx] == 'Left':
        left()
        # print('left')
    elif classes[idx] == 'Right':
        right()
        # print('right')

    cv2.imshow('result', img)
    # 아래 waitKey 꼭 해줘야함 그래야 잘 읽힘
    if cv2.waitKey(1) == ord('q'):  # q누르면 빠져나온다
        break
