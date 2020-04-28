from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

# Set Seed Value
np.random.seed(3)
tf.random.set_seed(3)

# Input Data
df = pd.read_csv('iris.csv',
                 names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

# Checked by Graph
sns.pairplot(df, hue='species');
plt.show()

# Classified by Data
dataset = df.values
X = dataset[:, 0:4].astype(float)
Y_obj = dataset[:,4]

# Convert string by Number
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y)

# Set Model
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Execute Model
model.fit(X, Y_encoded, epochs=50, batch_size=1)

# Print Output
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))
