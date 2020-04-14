import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]

x_data = [i[0] for i in data]       # 공부한 시간 데이터
y_data = [i[1] for i in data]       # 합격 여부

plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)
plt.show()

a = 0
b = 0

lr = 0.05       # 학습률


def sigmoid(x):                         # sigmoid라는 이름의 함수 정의
    return 1 / (1 + np.e ** (-x))       # 시그모이드 식의 형태 그대로 파이썬으로 옮김


# 경사 하강법
# 1000번 반복될 때 마다 x_data 값에 대한 현재의 a, b 값 출력
for i in range(2001):
    for x_data, y_data in data:
        # a에 대한 편미분
        a_diff = x_data * (sigmoid(a * x_data + b) - y_data)
        # b에 대한 편미분
        b_diff = sigmoid(a * x_data + b) - y_data
        # a, b 값 업데이트
        a = a - lr * a_diff
        b = b - lr * b_diff
        if i % 1000 == 0:
            print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))

plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)
x_range = (np.arange(0, 15, 0.1))    # 그래프로 나타낼 x 값의 범위 정하기
plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a * x + b) for x in x_range]))
plt.show()