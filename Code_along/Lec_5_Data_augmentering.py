import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

"""
Data augmentation
artificiellt skapa fler bilder
    slumpmässigt:
    roterar till en viss grad (radianer)
    translatera slumpmässigt
    flippa horisontellt, vertikalt (spegla)
    shear (skjuvning)
    ...
"""
(X_train, y_train), (X_test, y_test) = load_data()
y_train, y_test = y_train.ravel(), y_test.ravel()  # shapear om


def plot_samples(data):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(data[i], cmap="gray")
        ax.axis("off")
    plt.show()


#plot_samples(X_train)

# Scale data
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
print(X_train.min(), X_train.max())

# Train|val|test split
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=1 / 6, random_state=42)

# då ImageDataGenerator requires rank 4 , expanderar vi dimensionerna med den saknade colorchannel och ger None där
X_train_val = X_train_val[:, :, :, None]
X_train = X_train[:, :, :, None]
X_val = X_val[:, :, :, None]
X_test = X_test[:, :, :, None]

print(X_train_val.shape)

train_image_generator = ImageDataGenerator(rotation_range=10, shear_range=.2, zoom_range=.1, horizontal_flip=False,
                                           height_shift_range=.2, width_shift_range=.2)

# don't augment validation and test data
test_image_generator = ImageDataGenerator()

train_val_generator = train_image_generator.flow(X_train_val, y_train_val, batch_size=32)

val_generator = test_image_generator.flow(X_val, y_val, batch_size=32)

print(train_val_generator, val_generator)

print(len(train_val_generator.next()))
sample_batch = train_val_generator.next()
print(sample_batch[0].shape)  # 32 samples in a batch

plot_samples(sample_batch[0])
print(sample_batch[1])


# CNN model
def CNN_model(learning_rate=.001, drop_rate=.5, kernels=[32, 32]):
    adam = Adam(learning_rate=learning_rate)

    model = Sequential(name="CNN_model")

    # the convolutional layers
    for number_kernel in kernels:
                    conv_layer = Conv2D(number_kernel, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal",
                                        input_shape=X_train.shape[1:])

                    model.add(conv_layer)
                    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # MLP layers
    model.add(Flatten())
    model.add(Dropout(drop_rate))
    model.add(Dense(256, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer=adam, metrics=["acc"])

    return model


model = CNN_model(drop_rate=.5)
model.summary()

# Train on augmented data
steps_per_epoch = int(len(X_train_val) / 32)
validation_steps = int(len(X_val) / 32)

print(steps_per_epoch, validation_steps)

early_stopper = EarlyStopping(monitor="val_acc", mode="max", patience=5, restore_best_weights=True)

model.fit(train_val_generator, steps_per_epoch=steps_per_epoch, epochs=100, callbacks=[early_stopper],
          validation_data=val_generator, validation_steps=validation_steps)

metrics = pd.DataFrame(model.history.history)


def plot_metrics(metrics):
    _, ax = plt.subplots(1, 2, figsize=(12, 4))
    metrics[["loss", "val_loss"]].plot(ax=ax[0], grid=True)
    metrics[["acc", "val_acc"]].plot(ax=ax[1], grid=True)


plot_metrics(metrics)

# Train on all training data
train_generator = train_image_generator.flow(X_train, y_train, batch_size=32)

model = CNN_model()
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=15)

# Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
