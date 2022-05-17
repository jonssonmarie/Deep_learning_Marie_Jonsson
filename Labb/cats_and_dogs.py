import pandas as pd
import numpy as np
import cv2
import os, random, shutil, glob
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# import from other scripts in Labb folder
from plot_images import display_images, count_plot, joint_plot, check_if_random_plot
from create_dir import create_directory


os.chdir(os.path.dirname(__file__))
current_dir = os.path.abspath("")
print(f"{current_dir=}")

# list all files in current dir
files_current_dir = os.listdir(current_dir)
print(f"{files_current_dir=}")

# return absolut path
original_data_train_dir = os.path.abspath("original_data/train/train")
original_data_test_dir = os.path.abspath("original_data/test/test")

# list all files in original_data_train_dir
files_current_dir = os.listdir(original_data_train_dir)

# collect 10 images and print them to see if they are random
ten_train_images = random.sample(os.listdir(original_data_train_dir), k=10)
ten_test_images = random.sample(os.listdir(original_data_test_dir), k=10)
print(ten_train_images)
print(ten_test_images)

# 10 plots from Original Train and Original Test
display_images(ten_train_images, title="10 Original Train images", get_labels=False, path=original_data_train_dir)
display_images(ten_test_images, title="10 Original Test images", get_labels=False, path=original_data_test_dir)


create_directory(current_dir)

random.seed(142)
image_small_data = random.sample(os.listdir(original_data_train_dir), k=5000)

# sort in cats and dogs for saving to path according to amount
cats = [image for image in image_small_data if image[:3] == 'cat']
dogs = [image for image in image_small_data if image[:3] == 'dog']


def save_data_to_folder(image_lst):
    train_lst = []
    val_lst = []
    test_lst = []

    for data in image_lst:
        train_lst.extend(data[:800])
        val_lst.extend(data[800:1000])
        test_lst.extend(data[1000:1250])

    def copy_to_folder(file_lst, folder):  # fick tips från Felix om glob.glob
        if len(os.listdir(folder)) != 0:
            files = glob.glob(folder+'/*.jpg')
            for f in files:
                try:
                    os.remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
                continue
        if len(os.listdir(folder)) == 0:
            for filename in file_lst:
                src = f"{original_data_train_dir}/{filename}"
                dst = f"{folder}/{filename}"
                shutil.copyfile(src, dst)

    copy_to_folder(train_lst, f'{current_dir}/experiment_small_data/train/')
    copy_to_folder(val_lst, f'{current_dir}/experiment_small_data/val/')
    copy_to_folder(test_lst, f'{current_dir}/experiment_small_data/test/')


save_data_to_folder([cats, dogs])


def label_or_one_hot_encode(img_path, get_encoded=False):
    """ Jag fick inspiration av Daniel och Felix, jag hade gjort två funktioner som gjorde respektive one_hot och label
    och i förra labben blev det fel då en return inte följde med automatiskt så jag ville få in dem automatiskt så
    jag inte tappade dom vilket lätt sker om manuellt inlagt.
    :param img_path: path
    :param get_encoded: Bool
    :return: bool or string
    """
    if img_path.split(".")[0][-3:] == 'cat':
        if get_encoded:
            return 1
        return 'cat'
    else:
        if get_encoded:
            return 0
        return 'dog'


# glob finds all the path names matching a specified pattern according to the rules used by the Unix shell
train_image = [(plt.imread(img_path), label_or_one_hot_encode(img_path)) for img_path in
               glob.glob(f'{current_dir}/experiment_small_data/train/*.jpg')]
test_image = [(plt.imread(img_path), label_or_one_hot_encode(img_path)) for img_path in
              glob.glob(f'{current_dir}/experiment_small_data/test/*.jpg')]
val_image = [(plt.imread(img_path), label_or_one_hot_encode(img_path)) for img_path in
             glob.glob(f'{current_dir}/experiment_small_data/val/*.jpg')]

random.shuffle(train_image)
random.shuffle(test_image)
random.shuffle(val_image)


display_images(val_image[:10], title="10 Val images - small data", path=False, get_labels=True)
display_images(test_image[:10], title="10 Test images - small data", path=False, get_labels=True)
display_images(train_image[:10], title="10 Train images - small data", path=False, get_labels=True)

# used for plot to check random order and count plot
val_encoded = [label_or_one_hot_encode(img[1], get_encoded=True) for img in val_image]
train_encoded = [label_or_one_hot_encode(img[1], get_encoded=True) for img in train_image]
test_encoded = [label_or_one_hot_encode(img[1], get_encoded=True) for img in test_image]

check_if_random_plot((train_encoded, test_encoded, val_encoded),
                     ("Train random plot", "Test random plot", "Validation random plot"))

count_plot(val_encoded, "Val_images")
count_plot(train_encoded, "Train_images")
count_plot(test_encoded, "Test_images")


train_image = [(plt.imread(img_path), label_or_one_hot_encode(img_path)) for img_path in
               glob.glob(f'{current_dir}/experiment_small_data/train/*.jpg')]

test_image = [(plt.imread(img_path), label_or_one_hot_encode(img_path)) for img_path in
              glob.glob(f'{current_dir}/experiment_small_data/test/*.jpg')]

val_image = [(plt.imread(img_path), label_or_one_hot_encode(img_path)) for img_path in
             glob.glob(f'{current_dir}/experiment_small_data/val/*.jpg')]


def get_size_of_images(img_lst):
    return [img[0].shape for img in img_lst]


img_sizes = pd.DataFrame(get_size_of_images(train_image) + get_size_of_images(test_image) +
                         get_size_of_images(val_image), columns=["height", "width", "color"])

print("Min:", img_sizes.min(), "\nMax:", img_sizes.max())

joint_plot(img_sizes)


def preprocessing_data(data):
    """
    :param data: np.array
    :return: np.array
    """
    resized_val_images = [(cv2.resize(img[0], (160, 160)), img[1]) for img in data]
    return resized_val_images


train_resized = preprocessing_data(train_image)
test_resized = preprocessing_data(test_image)
val_resized = preprocessing_data(val_image)

random.shuffle(train_resized)
random.shuffle(test_resized)
random.shuffle(val_resized)

display_images(train_resized[:10], title="Train resized - small data", get_labels=True, path=False)
display_images(test_resized[:10], title="Test resized - small data", get_labels=True, path=False)
display_images(val_resized[:10], title="Val resized - small data", get_labels=True, path=False)

# Train|val|test split and Normalize data
print()
scaled_X_train = np.array([image[0] for image in train_resized]).astype("float32") / 255.0 + 0.01
scaled_X_test = np.array([image[0] for image in test_resized]).astype("float32") / 255.0 + 0.01
scaled_X_val = np.array([image[0] for image in val_resized]).astype("float32") / 255.0 + 0.01
print("scaled_X", scaled_X_train.shape, scaled_X_test.shape, scaled_X_val.shape)
# adding 0.01. This way, we avoid 0 values as inputs, which are capable of preventing weight updates
# https://python-course.eu/machine-learning/training-and-testing-with-mnist.php

# Train|val|test split
y_train = np.array([label_or_one_hot_encode(image[1], get_encoded=True) for image in train_resized])
y_test = np.array([label_or_one_hot_encode(image[1], get_encoded=True) for image in test_resized])
y_val = np.array([label_or_one_hot_encode(image[1], get_encoded=True) for image in val_resized])
print("scaled_y and encoded_y", y_train.shape, y_test.shape, y_val.shape)

# concatenate of Train+val
scaled_X_train_val = np.concatenate((scaled_X_train, scaled_X_val), axis=0)
y_train_val = np.concatenate((y_train, y_val), axis=0)
print("scaled_X_train_val and y_train_val", scaled_X_train_val.shape, y_train_val.shape)


# Evaluation
def evaluation(model, x_for_pred, y_true, title):
    """
    :param model: model
    :param x_for_pred: np.array
    :param y_param: np.array
    :param title: string
    :return: None
    """
    y_pred = model.predict(x_for_pred)
    y_pred = (y_pred > .5) * 1

    print("Evaluation:", title)
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig('saved_images/' + f"confusion_matrix__{title}.png")

    misclassified_indices = np.where(y_pred.flatten != y_true)[0]
    misclassified_samples = x_for_pred[misclassified_indices]
    print("number of misclassified images:", cm[0][1] + cm[1][0])
    # plot some misclassified images
    random.shuffle(misclassified_samples)
    display_images(misclassified_samples, title="Misclassified", nrows=8, ncols=6, path=False, get_labels=False)


def plot_metrics(metrics, title):
    """
    :param metrics: np.array
    :param title: string
    :return: None
    """
    _, ax = plt.subplots(1, 2, figsize=(12, 4))
    metrics[["loss", "val_loss"]].plot(ax=ax[0], title=title, grid=True)
    metrics[["acc", "val_acc"]].plot(ax=ax[1], title=title, grid=True)
    plt.savefig(f"saved_images/metrics__{title}.png")


augmented_image_generator = ImageDataGenerator(rotation_range=10,
                                               shear_range=.1,
                                               zoom_range=.2,
                                               horizontal_flip=1,
                                               height_shift_range=.2,
                                               width_shift_range=.2,
                                               vertical_flip=0)

wo_augmented_image_generator = ImageDataGenerator()  # used for no augmentation

augmented_train_generator = augmented_image_generator.flow(scaled_X_train, y_train, batch_size=32)

wo_augmented_train_generator = wo_augmented_image_generator.flow(scaled_X_train, y_train, batch_size=32)

val_generator = wo_augmented_image_generator.flow(scaled_X_val, y_val, batch_size=32)

test_generator = wo_augmented_image_generator.flow(scaled_X_test, y_test, batch_size=32)


# For scaled_X_train_val, y_train_val = scaled_X_train + scaled_X_val,  y_train + y_val
train_val_generator = augmented_image_generator.flow(scaled_X_train_val, y_train_val, batch_size=32)
train_val_wo_generator = wo_augmented_image_generator.flow(scaled_X_train_val, y_train_val, batch_size=32)

sample_batch = augmented_train_generator.next()
display_images(sample_batch[0], title="augumented", path=False, get_labels=False)


input_shape = scaled_X_train.shape[1:]


def CNN_model(kernels, drop_rate, learn_rate):
    """
    :param kernels: int
    :param drop_rate: float
    :param learn_rate: float
    :return: model
    """
    model = Sequential(name="CNN_model")

    # the convolutional layers
    for number_kernel in kernels:
        conv_layer = Conv2D(number_kernel,
                            kernel_size=(3, 3),
                            activation="relu",
                            kernel_initializer="he_normal",
                            input_shape=input_shape)

        model.add(conv_layer)
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # MLP layers
    model.add(Flatten())
    model.add(Dropout(drop_rate))
    model.add(Dense(512, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learn_rate), metrics=["acc"])

    return model


model = CNN_model(kernels=[32, 64, 128, 128], drop_rate=.2, learn_rate=.001)
model.summary()

steps_per_epoch = int(len(scaled_X_train) / 32)
validation_steps = int(len(scaled_X_val) / 32)


# Train on augmented data
early_stopper = EarlyStopping(monitor="val_acc", mode="max", patience=15, restore_best_weights=True)

model.fit(augmented_train_generator, steps_per_epoch=steps_per_epoch, epochs=50, callbacks=[early_stopper],
          validation_data=val_generator, validation_steps=validation_steps)

metrics = pd.DataFrame(model.history.history)
plot_metrics(metrics, "Augmented_Training_X_train")
evaluation(model, scaled_X_val, y_val, "Augmented_Training_train_val")


# Train without augmented data
model.fit(wo_augmented_train_generator, steps_per_epoch=steps_per_epoch, epochs=50, callbacks=[early_stopper],
          validation_data=val_generator, validation_steps=validation_steps)
model.save('saved_models/train_val_wo_augmented_generator.h5')
metrics_wo = pd.DataFrame(model.history.history)
plot_metrics(metrics_wo, "wo_augmented_Training")
evaluation(model, scaled_X_val, y_val, "wo_augmented_Training_train_val")


# Train+val
steps_per_epoch_all = int(len(scaled_X_train_val) / 32)
model_all = CNN_model(kernels=[32, 64, 128, 128], drop_rate=.3, learn_rate=.001)
model_all.summary()

# Train on train+val on augmented data
model_all.fit(train_val_generator, steps_per_epoch=steps_per_epoch_all, epochs=10)
evaluation(model_all, scaled_X_test, y_test, "Augmented_Training_train_+_val__test")

# Train on train+val without augmented data
model_all.fit(train_val_wo_generator, steps_per_epoch=steps_per_epoch_all, epochs=10)
evaluation(model_all, scaled_X_test, y_test, "Wo_augmented_Training_train_+_val__test")


# VGG16
steps_per_epoch_vvg16 = int(len(scaled_X_train) / 32)
validation_steps_vgg16 = int(len(scaled_X_val) / 32)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
base_model.summary()

vgg16_model = Sequential([base_model,
                          Flatten(),
                          Dropout(0.2),
                          Dense(512, activation="relu", kernel_initializer="he_normal"),
                          Dense(1, activation="sigmoid")],
                         name="vgg16_model")

base_model.trainable = False
set_trainable = False

vgg16_model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-5), metrics=['acc'])
vgg16_model.summary()

es = EarlyStopping(monitor="val_acc", mode="max", patience=10, restore_best_weights=True)
vgg16_model.fit(wo_augmented_train_generator,
                steps_per_epoch=steps_per_epoch_vvg16,
                epochs=15,
                callbacks=es,
                validation_data=val_generator,
                validation_steps=validation_steps_vgg16)

metrics = pd.DataFrame(vgg16_model.history.history)
plot_metrics(metrics, "vgg16")
evaluation(vgg16_model, scaled_X_val, y_val, "vgg16_wo_Augmented_Tuning_train__val")
