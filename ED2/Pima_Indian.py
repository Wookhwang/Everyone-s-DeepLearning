"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 판다스를 이용해 데이터 불러오기 및 데이터 칼럼의 데이터 명 입력
df = pd.read_csv('pima-indians-diabetes.csv',
                 names=["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])


print(df.head(5))                       # 데이터 5줄을 불러오기
print(df.info())                        # 데이터의 전반적인 정보 확인
print(df.describe())                    # 데이터의 자세한 정보 확인
print(df[['pregnant', 'class']])        # 일부 컬럼의 정보만 확인

# 데이터 가공하기
print(df[['pregnant', 'class']].groupby(['pregnant'],
                                        as_index=False).mean().sort_values(by='pregnant', ascending=True))

# matplotlib를 이용해 그래프로 표현하기
plt.figure(figsize=(12, 12))
# heatmap()를 통해 그래프를 표현
sns.heatmap(df.corr(),
            linewidths=0.1,             # 선 넓이
            vmax=0.5,                   # 색상의 밝기를 조절하는 인자
            cmap=plt.cm.gist_heat,      # 미리 정해진 matplotlib 색상의 설정값을 불러옴
            linecolor='white',          # 선 색상
            annot=True)
plt.show()

# 그래프의 결과로 class와 plasma의 상관관계가 가장 높음을 파악
# class와 plasma를 따로 떼어 두 항목 간의 관계를 그레프로 재확인
grid = sns.FacetGrid(df, col='class')
grid.map(plt.hist, 'plasma', bins=10)
plt.show()
# class=1 인 경우에 plasma가 150 이상인 환자가 많다는 결론 도출 가능
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 예측 실행
# seed 값 생성
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 로드
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

# 모델의 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
model.fit(X, Y, epochs=200, batch_size=10)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
