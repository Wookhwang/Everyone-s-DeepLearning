import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as statm
import statsmodels.formula.api as statfa
from mpl_toolkits import mplot3d    # 3D 그래프 그리는 라이브러리

# 공부 시간 X와 성적 Y의 리스트 만들기
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data]   # 학습시간
x2 = [i[1] for i in data]   # 과외시간
y = [i[2] for i in data]    # 성적

# 그래프로 확인
ax = plt.axes(projection='3d')  # 그래프 유형 3D로 정하기
ax.set_xlabel('Study_Hours')
ax.set_ylabel('Private_Class')
ax.set_zlabel('Score')
ax.dist = 11
ax.scatter(x1, x2, y)
plt.show()

# 리스트로 되어 있는 x와 y값을 넘파이 배열로 바꾸기
# (인덱스를 주어 하나씩 불러와 계산이 가능하게 하기 위함)
x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

# 기울기 a와 절편 b의 값을 초기화
a1 = 0
a2 = 0
b = 0

# 학습률
lr = 0.05

# 몇 번 반복할지 설정(0부터 세므로 원하는 반복 횟수에 +1)
epochs = 2001

# 경사 하강법 시작
for i in range(epochs):                         # epochs 수 만큼 반복
    y_pred = x1_data * a1 + x2_data * a2 + b    # y를 구하는 식 세우기
    error = y_data - y_pred                     # 오차를 구하는 식
    # 오차 함수를 a1으로 미분한 값
    a1_diff = -(1 / len(x1_data)) * sum(x1_data * error)
    # 오차 함수를 a2으로 미분한 값
    a2_diff = -(1 / len(x2_data)) * sum(x2_data * error)
    # 오차 함수를 b로 미분한 값
    b_diff = -(1 / len(x1_data)) * sum(y_data - y_pred)
    a1 = a1 - lr * a1_diff                      # 학습률을 통해 기존의 a1 value update
    a2 = a2 - lr * a2_diff                      # 학습률을 통해 기존의 a1 value update
    b = b - lr * b_diff                         # 학습률을 통해 기존의 a1 value update

    if i % 100 == 0:
        print("epochs=%.f, 기울기1=%.04f, 기울기2=%.04f, 절편=%.04f" % (i, a1, a2, b))


X = [i[0:2] for i in data]
y = [i[2] for i in data]

X_1=statm.add_constant(X)
results=statm.OLS(y,X_1).fit()

hour_class=pd.DataFrame(X,columns=['study_hours','private_class'])
hour_class['Score']=pd.Series(y)

model = statfa.ols(formula='Score ~ study_hours + private_class', data=hour_class)

results_formula = model.fit()

a, b = np.meshgrid(np.linspace(hour_class.study_hours.min(),hour_class.study_hours.max(),100),
                   np.linspace(hour_class.private_class.min(),hour_class.private_class.max(),100))

X_ax = pd.DataFrame({'study_hours': a.ravel(), 'private_class': b.ravel()})
fittedY=results_formula.predict(exog=X_ax)

fig = plt.figure()
graph = fig.add_subplot(111, projection='3d')

graph.scatter(hour_class['study_hours'],hour_class['private_class'],hour_class['Score'],
              c='blue',marker='o', alpha=1)
graph.plot_surface(a,b,fittedY.values.reshape(a.shape),
                   rstride=1, cstride=1, color='none', alpha=0.4)
graph.set_xlabel('study hours')
graph.set_ylabel('private class')
graph.set_zlabel('Score')
graph.dist = 11

plt.show()
