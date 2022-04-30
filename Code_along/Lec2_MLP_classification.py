
# MLP for image classification

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.datasets.mnist import load_data
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

(X_train, y_train), (X_test, y_test) = load_data()
print(X_train.shape, X_test.shape)

# Normalize data X-scaled = X - X_min / X_max - X_min
print(f"min: {X_train.min()}, max: {X_train.max()}")

X_train = X_train.astype("float32")/255
X_test = X_test.astype("float32")/255
X_test.min(), X_test.max()


def display_images(data, nrows=2, ncols=5, figsize=(12, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize = figsize)

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(data[i, :, :], cmap="gray")
        ax.axis("off")

    fig.subplots_adjust(wspace=0, hspace=.1, bottom=0)
    plt.show()


display_images(X_train)


# MLP model
def MLP_model(nodes=None, names=None, activations=[]):
    model = Sequential(name="MLP_model")
    model.add(Flatten(input_shape=(28, 28), name="input_layer"))

    for node, name, activation in zip(nodes, names, activations):
        model.add(Dense(node, name=name, activation=activation))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


model_1 = MLP_model(nodes=[10], names=["Output_layer"], activations=["softmax"])
model_1.summary()
# 28x28 -> 784 -> 784 weights & 1 bias -> 785 * 10 (output nodes) -> 7850

model_1.fit(X_train, y_train, validation_split=1/6, epochs=20, verbose=0)

metrics = pd.DataFrame(model_1.history.history)
metrics.index = range(len(metrics))
metrics.head()


def plot_metrics(df_history, style="-o"):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    columns = [["loss", "val_loss"], ["accuracy", "val_accuracy"]]
    for ax, col in zip(axes, columns):
        df_history.plot(y=col, xlabel="Epochs",
                        ylabel=col[0], ax=ax, style=style)
        ax.grid()
    plt.show()


plot_metrics(metrics)

# Hidden layers
model_2 = MLP_model(nodes=[128, 128, 10], activations=[
                    "relu", "relu", "softmax"], names=["Hidden1", "Hidden2", "Output"])
model_2.summary()
model_2.fit(X_train, y_train, validation_split=1/6, epochs=20, verbose=1)

metrics = pd.DataFrame(model_2.history.history)
metrics.index = range(len(metrics))

plot_metrics(metrics, style = "-")

model_3 = MLP_model(nodes=[128, 128, 10], activations=[
                    "relu", "relu", "softmax"], names=["Hidden1", "Hidden2", "Output"])
model_3.fit(X_train, y_train, epochs = 5, verbose = 1)  # manual early stopping, and train on all training data


# Prediction and evaluation
y_pred = model_3.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)  # plockar ut det största värdet längs axis 1

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()


# Misclassification Kollar vilka bilder som blir dåliga
misclassified_indices = np.where(y_pred != y_test)
misclassified_samples = X_test[misclassified_indices]

display_images(misclassified_samples, 4, 5, (12, 8))
