import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공부 시간 X와 성적 Y의 리스트를 만들기
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# 그래프로 나타내기
plt.figure(figsize=(8, 5))
plt.scatter(x, y)
plt.show()

# 리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸기
# (인덱스를 주어 하나씩 불러와 계산이 가능하게 하기 위함)
x_data = np.array(x)
y_data = np.array(y)

# 기울기 a와 절편 b의 값 초기화
a = 0
b = 0

# 학습률 정하기
lr = 0.05

# 몇 번 반복될지 설정(0부터 세므로 원하는 반복 횟수에 +1)
epochs = 2001

# 경사 하강법 시작
for i in range(epochs):         # 에포크 수 만큼 반복
    y_pred = a * x_data + b     # 예측할 y 값을 구하는 식 세우기
    error = y_data - y_pred     # 오차를 구하는 식
    # 오차 함수를 a로 미분한 값
    a_diff = -(1 / len(x_data)) * sum(x_data * error)
    # 오차 함수를 b로 미분한 값
    b_diff = -(1 / len(x_data)) * sum(y_data - y_pred)

    a = a - lr * a_diff         # 학습률을 곱해 기존의 a값을 업데이트
    b = b - lr * b_diff         # 학습률을 곱해 기존의 b값을 업데이트

    if i % 100 == 0:            # 100번 반복될 때마다 현재의 a값, b값 출력
        print("epochs=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))

# 앞서 구한 기울기와 절편을 이용해 그래프를 다시 그리기
y_pred = a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()



