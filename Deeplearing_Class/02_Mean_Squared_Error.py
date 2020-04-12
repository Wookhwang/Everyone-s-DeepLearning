import numpy as np

# 기울기와 y 절편
fake_a_b = [3, 76]

# x, y 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]


# y = ax + b에 a와 b의 값을 대입하여 결과를 출력하는 함수
def predict(x):
    return fake_a_b[0] * x + fake_a_b[1]


# MSE 함수
def mse(y_hat, y):
    return ((y_hat - y) ** 2).mean()


# MSE 함수를 각 y 값에 대입하여 최종 값을 구하는 함수
def mse_value(predict_result, y):
    return mse(np.array(predict_result), np.array(y))


# 예측 값이 들어갈 빈 리스트
predict_result = []


# 모든 x값을 한 번씩 대입
for i in range(len(x)):
    # predict_result 에 대입
    predict_result.append(predict(x[i]))
    print("공부한 시간=%.f, 실제 점수=%.f, 예측 점수=%.f" % (x[i], y[i], predict(x[i])))


# 최종 MSE 출력
print("최종 MSE 깂 :" + str(mse_value(predict_result, y)))
