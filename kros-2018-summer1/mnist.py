import numpy as np
from keras import datasets        #mnist data를 불러오기 위한 library
from keras.utils import np_utils  #to_categorical 불러오기 위한 library


def Data_function():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data() #train data와 test data 각각 불러오기
    
    Y_train = np_utils.to_categorical(y_train) # to_categorical() : 0-9 사이의 출력값 -> 이진벡터 (분류 작업 효율이 더 좋음)
                                               # Ex) [1 9] -> [[0 1 0 0 0 0 0 0 0 0],[0 0 0 0 0 0 0 0 0 1]]
    Y_test = np_utils.to_categorical(y_test)
    L, W, H = X_train.shape            # L: sample의 개수 60000개, W: width = 28, H: height = 28
    X_train = X_train.reshape(-1, W*H) # [28x28] matrix를 1 by 784 인 matrix로 reshape 
                                       # -1을 row의 input으로 넣으면 열에 대응하는 row의 개수가 자동으로 입력
    print(X_train[1].shape)
    X_test = X_test.reshape(-1, W*H)
    
    X_train = X_train / 255.0          # 0에서 255 사이의 값을 가지는 element를 0에서 1 사이의 실수로 normalize 시키는 과정
    X_test = X_test / 255.0
    
    return (X_train, Y_train), (X_test, Y_test)
