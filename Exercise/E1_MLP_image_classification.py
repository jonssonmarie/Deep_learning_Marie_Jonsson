import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Flatten, Dense, InputLayer, Activation, Dropout
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.datasets.fashion_mnist import load_data
# Multilayered perceptron (MLP) image classification exercises
"""
0. Fashion dataset (*)
Zalando has provided an MNIST dataset for fashion, with the format very similar to the original MNIST digits dataset. 
Start with loading this fashion dataset from TensorFlow Keras.
  a) Start visualizing some of the sample images
  b) Normalize the images to values between 0 and 1
  c) Visualize same images as before, do you see any difference?
  d) Make histogram of a sample image before and after normalization. What do you notice?
  e) Check if the dataset is balanced.
"""

(X_train, y_train), (X_test, y_test) = load_data()

# Normalize data  X_scaled = X - X_min / X_max - X_min
print(f" min: {X_train.min()}, max: {X_train.max()}")
X_train = X_train.astype("float32")/255  # snabbare beräkning med 32 bit
X_test = X_test.astype("float32")/255
print(f" min: {X_train.min()}, max: {X_train.max()}")


def display_images(data, nrows=2, ncols=5, figsize=(12, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(data[i,:,:], cmap="gray")
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

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


model_naive = MLP_model(nodes=[10], names=["Hidden1", "Hidden2", "Output_layer"], activations=["softmax"])
model_naive.summary()


model_naive.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, verbose=0)

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

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

y_pred = model_deep.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()

misclassified_indices = np.where(y_pred != y_test)[0]
misclassified_samples = X_test[misclassified_indices]

# a few misclassifications
display_images(misclassified_samples, 4,5, (12,8))

"""

Lyckades precis få 0.9014 😄 Använde 2 hidden layers med 512 neuroner i det första och 512 neuroner i det andra. 
Sen bytte jag optimizer = adam till adamax efter att ha läst att den lyckas bättre än Adam i vissa fall. 
epoker = 50

något i 1.1 a
_, _,_, y_test_test = train_test_split(X_train, y_train, test_size=10000)
unique, counts = np.unique(y_test_test, return_counts= True)
occurences = dict(zip(unique, counts))
print(occurences)

"""


