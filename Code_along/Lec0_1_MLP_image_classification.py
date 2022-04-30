import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# annorlunda uppdelning här direkt från keras inte vanliga train_test_split
(X_train, y_train), (X_test, y_test) = load_data("mnist.npz")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Normalize data  X_scaled = X - X_min / X_max - X_min
print(f" min: {X_train.min()}, max: {X_train.max()}")
X_train = X_train.astype("float32")/255  # snabbare beräkning med 32 bit
X_test = X_test.astype("float32")/255
print(f" min: {X_train.min()}, max: {X_train.max()}")


def display_images(data, nrows = 2, ncols = 5, figsize = (12,4)):
    fig, axes = plt.subplots(nrows, ncols, figsize = figsize)

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(data[i,:,:], cmap = "gray")
        ax.axis("off")

    fig.subplots_adjust(wspace=0, hspace=.1, bottom=0)
    plt.show()


display_images(X_train)


sns.displot(data=X_train[5000, :, :].reshape(-1), kind="hist", legend=True)
plt.show()


def MLP_model(nodes=None, names=None, activations=[]):
    model = Sequential(name="MLP_model")
    # flattens the input
    model.add(Flatten(input_shape=(28, 28), name="Input_layer"))

    for node, name, activation in zip(nodes, names, activations):
        model.add(Dense(node, name=name, activation=activation))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    return model


model_naive = MLP_model(nodes=[10], names=["Output_layer"], activations=["softmax"])
model_naive.summary()

model_naive.fit(X_train, y_train, validation_split=1/6, epochs=20, verbose=0)

naive_history = pd.DataFrame(model_naive.history.history)
naive_history.index = range(len(naive_history))
naive_history.head()


def plot_metrics(df_history, style="-o"):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    metrics = [["loss", "val_loss"], ["accuracy", "val_accuracy"]]
    for ax, metric in zip(axes, metrics):
        df_history.plot(y=metric, xlabel="Epochs",
                        ylabel=metric[0],
                        title=metric[0], ax=ax, style=style)
    plt.show()


plot_metrics(naive_history)
# don't let the scale on y-axis trick you, the curves are in fact very close to each other

# Hidden layers
model_deep = MLP_model(nodes=[128, 128, 10], activations=["relu", "relu", "softmax"], names=["Hidden_1", "Hidden_2", "Output"])

model_deep.summary()

model_deep.fit(X_train, y_train, validation_split=1/6, verbose=0, epochs=20)

deep_history = pd.DataFrame(model_deep.history.history)
deep_history.index = range(len(deep_history))
deep_history.head()


plot_metrics(deep_history, style = "-")
# we see validation loss increases and overfits
# validation accuracy has stabilized, however in severe overfitting
# the model fits to a lot of noise in training data could lead to significant
# drop in validation accuracy

model_deep.fit(X_train, y_train, epochs = 8, verbose=0)



y_pred = model_deep.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()

misclassified_indices = np.where(y_pred != y_test)[0]
misclassified_samples = X_test[misclassified_indices]

# a few misclassifications
display_images(misclassified_samples, 4,5, (12,8))


