from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

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

# Spilt into File
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

# Empty Accuracy Array
accuracy = []

# Set Model, Compile, Execute
for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.fit(X[train], Y[train], epochs=100, batch_size=5)

    K_accuracy = "% .4f" % (model.evaluate(X[test], Y[test])[1])
    accuracy.append(K_accuracy)

# Print Output
print("\n %.f Fold Accuracy" % n_fold, accuracy)