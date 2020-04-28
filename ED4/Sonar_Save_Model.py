from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import pandas as pd

# Set Seed Value
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)
df = pd.read_csv('Sonar.csv', header=None)

# Set dataset
dataset = df.values
X = dataset[:, 0:60]
Y_obj = dataset[:, 60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# Divided Train Set & Test Set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=130, batch_size=5)
model.save('Sonar_Model.h5')    # 모델을 컴퓨터에 저장

del model   # 기존 메모리 내의 모델 삭제
model = load_model('Sonar_Model.h5')    # 모델을 새로 불러옴

# Apply Model to Test Set
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))