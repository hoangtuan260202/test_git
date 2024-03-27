#Thêm thư viện 
from numpy import expand_dims
import numpy as np
import cv2
# from keras.datasets.cifar10 import load_data
from keras.datasets.fashion_mnist import load_data


def load_real_samples():
    # load dataset
    (trainX, trainy), (_, _) = load_data()
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return [X, trainy]

# # load dataset
# (trainX, trainy), (_, _) = load_data()

# # Chuyển ảnh thành kích thước 28x28
# num_samples = trainX.shape[0]

# resized_images = np.zeros((num_samples, 28, 28, 1))  # Tạo mảng mới chứa ảnh kích thước 28x28

# for i in range(num_samples):
#     gray_image = cv2.cvtColor(trainX[i], cv2.COLOR_BGR2GRAY)
#     # Resize ảnh
#     resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)

#     # Mở rộng chiều cuối cùng để tạo thành ảnh màu
#     X = expand_dims(resized_images, axis=-1)

#     # Chuyển đổi từ ints sang floats
#     X = X.astype('float32')

#     # Chuẩn hóa về khoảng [-1, 1]
#     X = (X - 127.5) / 127.5
