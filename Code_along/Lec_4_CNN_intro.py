import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets.cifar10 import load_data
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

(X_train, y_train), (X_test, y_test) = load_data()
y_train, y_test = y_train.ravel(), y_test.ravel()
# ravel  generell 50000,1 [50000,1] till 50000 [50000] - >gör 2D till 1D

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

labels_dict = {i: label for i, label in enumerate(labels)}

print(f"{X_train.shape=}, {X_test.shape=}\n{y_train.shape=}, {y_test.shape=}")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_train[i])
    ax.axis("off")
    ax.set_title(f"{y_train[i]}: {labels[y_train[i]]}")
fig.tight_layout()

print(f"{X_train.min()=}, {X_train.max()=}")

# Skalar datan
scaled_X_train = X_train.astype("float32") / 255
scaled_X_test = X_test.astype("float32") / 255


def CNN_model(learning_rate=0.001, drop_rate=0.5, kernels=[32, 64]):
    print(drop_rate)
    # default learning rate in Adam
    adam = Adam(learning_rate=learning_rate)

    model = Sequential(name="CNN_model")

    # convolutional layers
    for number_kernel in kernels:
        conv_layer = Conv2D(
            number_kernel,
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=X_train.shape[1:])  # (ger 32,32,32 tar bort 50000 iom 1:)

        model.add(conv_layer)
        # defaults to pool_size if None
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())
    model.add(Dropout(drop_rate))
    model.add(Dense(256, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(10, activation="softmax"))  # varje nod representerar en label

    model.compile(loss="sparse_categorical_crossentropy", optimizer=adam, metrics=["acc"])  # acc = accuracy

    return model


model = CNN_model(.001, .5, [32, 64, 32])

print(np.unique(y_train))
model.summary()

early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=5)  # 15 , men 5 för snabbare körning

model.fit(scaled_X_train, y_train, epochs=100, callbacks=[early_stopper], validation_split=1 / 5)


def plot_metrics(metrics):
    _, ax = plt.subplots(1, 2, figsize=(12, 4))
    metrics[["loss", "val_loss"]].plot(ax=ax[0], title="Loss", grid=True)
    metrics[["acc", "val_acc"]].plot(ax=ax[1], title="Accuracy", grid=True)


metrics = pd.DataFrame(model.history.history)
plot_metrics(metrics)

# Hyperparameter tuning, tunar 1 parameter här
early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)


def evaluate_model(**kwargs):
    model = CNN_model(**kwargs)
    model.fit(scaled_X_train, y_train, validation_split=1 / 6, epochs=6, callbacks=[early_stopper])
    # small epochs to make training faster 16 men ändrat till 6 för snabbare körning
    metrics = pd.DataFrame(model.history.history)
    val_acc = metrics["val_acc"].iloc[-1]  # -1 tar sista värdet
    return val_acc


dropout_accuracies = {}
for drop_rate in np.arange(.1, .6, .1):
    # round because of floating point precision
    drop_rate = np.round(drop_rate, 1)
    dropout_accuracies[drop_rate] = evaluate_model(drop_rate=drop_rate)

pd.DataFrame(dropout_accuracies.values(), index=dropout_accuracies.keys()). \
    plot(xlabel="Dropouts", ylabel="validation_acc", style="--o")

# Train and evaluate on chosen model
# note that we can't use early stopping here as we will train on all training data and no validation
# don't use test data as validation data here

model_final = CNN_model(drop_rate=.4)
model_final.fit(scaled_X_train, y_train, epochs=20)

y_pred = model.predict(scaled_X_test)
# y_pred = np.argmax(y_pred)

y_pred = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=labels).plot()
plt.xticks(rotation=90)
