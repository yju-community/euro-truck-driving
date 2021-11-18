import numpy as np
from Modules.alexnet import alexnet  # CNN Convolution Neural Network

VERSION = 5

WIDTH = 256
HEIGHT = 160
LR = 1e-3
EPOCHS = 10
MODEL_NAME = '../Models/euro-truck-fast-{}-{}-{}-epochs-300K-data.model'.format(
    LR, 'alexnetv2', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

for i in range(EPOCHS):
    for i in range(1, VERSION+1):
        train_data = np.load(
            '../Dataset/training_data_v{}_balanced.npy'.format(i), allow_pickle=True)

        # 원하는 크기로 train, test 데이터를 나눔
        train = train_data[:-100]
        test = train_data[-100:]

        # 데이터(screen)과 정답(out_put)을 분리
        X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)


# tensorboard --logdir=

# loss: 훈련 손실값
# acc: 훈련 정확도
# val_loss: 검증 손실값
# val_acc: 검증 정확도
