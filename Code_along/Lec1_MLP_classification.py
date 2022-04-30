# MLP classification code along

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

raw_data = load_breast_cancer()
X, y = raw_data.data, raw_data.target

print(f"Any nans? {np.isnan(X).any()}")
print(X.shape, y.shape)

# Train | Test split
scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# MLP network
def MLP():
    model = Sequential(name = "MLP")
    model.add(InputLayer(X.shape[1], name = "Input_layer"))
    model.add(Dense(32, name = "Hidden1", activation = "relu")) # change to he initializer
    model.add(Dense(32, name = "Hidden2", activation = "relu"))
    model.add(Dense(1, name = "Output", activation = "sigmoid"))

    model.compile(loss = "binary_crossentropy", optimizer = "adam")
    return model


print(f"Training parameters {(30+1)*32+(33*32)+33}")
model = MLP()
print(model.summary())

model.fit(scaled_X_train, y_train, epochs = 500, validation_split=.2, verbose=1)


df_loss = pd.DataFrame(model.history.history)
print(df_loss.head())

df_loss.plot()
# clear overfitting as validation loss increases after a certain amount of epochs

# Early Stopping
model = MLP()
print(model.summary())
model.fit(scaled_X_train, y_train, epochs = 50, validation_split=.2, verbose=1)
pd.DataFrame(model.history.history).plot()
model = MLP()
model.fit(scaled_X_train, y_train, epochs = 50, verbose=0)

# prediction and evaluation
y_pred = model.predict(scaled_X_test)
y_pred = (y_pred > .5)*1

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()
print(classification_report(y_test, y_pred))
