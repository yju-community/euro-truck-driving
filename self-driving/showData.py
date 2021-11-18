import numpy as np
import cv2

VERSION = 5

# train_data = np.load('../Dataset/training_data_v{}_balanced.npy'.format(VERSION), allow_pickle=True)
train_data = np.load(
    '../Dataset/training_data_v{}.npy'.format(VERSION), allow_pickle=True)

for data in train_data:
    img = data[0]
    choice = data[1]

    cv2.imshow('test', img)
    print(choice)

    # exit 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
