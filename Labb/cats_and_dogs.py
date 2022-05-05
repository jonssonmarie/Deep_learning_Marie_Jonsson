import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import os, random, shutil, glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Flatten, Dense, Conv2D, InputLayer, Activation, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
"""from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"""
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping

os.chdir(os.path.dirname(__file__))

current_dir = os.path.abspath("")
print(f"{current_dir=}")

# list all files in current dir
files_current_dir = os.listdir(current_dir)
print(f"{files_current_dir=}")

original_data_train_dir = os.path.abspath("original_data/train/train")
original_data_test_dir = os.path.abspath("original_data/test/test")

files_current_dir = os.listdir(original_data_train_dir)

ten_train_images = random.sample(os.listdir(original_data_train_dir), k=10)
ten_test_images = random.sample(os.listdir(original_data_test_dir), k=10)
#print(ten_train_images)
#print(ten_test_images)


def display_images(data, path, title=None, nrows=2, ncols=5, figsize=(15, 8)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for pict, ax in zip(data, axes.flatten()):
        #img = plt.imread(os.path.join(path, pict))
        img = plt.imread(f"{path}/{pict}")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(pict, fontsize=12)
        fig.suptitle(title)

    fig.subplots_adjust(wspace=0, hspace=.1, bottom=0)
    plt.show()


#display_images(ten_train_images, original_data_train_dir, "10 Train images")
#display_images(ten_test_images, original_data_test_dir, "10 Test images")


def create_directory():
    top_folders = ["experiment_small_data", "experiment_tiny"]  # experiment_tiny
    sub_folders = ["test", "train", "val"]
    try:
        for top_folder in top_folders:
            top_path = os.path.join(current_dir, top_folder)
            os.mkdir(top_path)
            for sub_folder in sub_folders:
                sub_path = os.path.join(top_folder, sub_folder)
                os.mkdir(sub_path)
    except:
        print("Folder exist already")
        pass


create_directory()

"""
d) Nu ska du göra train|val|test split med följande splits:
experiment_small
"""
# random.sample - Used for random sampling without replacement.
random.seed(42)
image_small_data = random.sample(os.listdir(original_data_train_dir), k=5000)

# sort in cats and dogs for saving to path according to amount
cats = [image for image in image_small_data if image[:3] == 'cat']
dogs = [image for image in image_small_data if image[:3] == 'dog']


# {}, kollade nu left shift + right option + 8/9
def save_data_to_folder(image_lst):
    train_lst = image_lst[:800]
    val_lst = image_lst[800:1000]
    test_lst = image_lst[1000:1250]

    def copy_to_folder(file_lst, folder):
        for filename in file_lst:
            src = f"{original_data_train_dir}/{filename}"
            dst = f"{folder}/{filename}"
            shutil.copyfile(src, dst)
    # testa om utan inre def, så som jag testade innan
    copy_to_folder(train_lst, f'{current_dir}/experiment_small_data/train/')
    copy_to_folder(val_lst, f'{current_dir}/experiment_small_data/val/')
    copy_to_folder(test_lst, f'{current_dir}/experiment_small_data/test/')


#save_data_to_folder(cats)
#save_data_to_folder(dogs)


"""
e) Läs in dataseten från experiment_small, experiment_tiny och plocka ut labelsvektorer, som ska vara
one-hot encoded med 0 och 1.
plotta några bilder med deras respektive labels och kontrollera att det är korrekt.
skapa lämplig plot för att kontrollera att dataseten är balanserade
skapa lämplig plot för att kontrollera att dataseten är slumpade (dvs inte ex [0, 0, ... 0, 1, 1, ..., 1]).
"""
train_path = os.path.abspath("experiment_small_data/train")
test_path = os.path.abspath("experiment_small_data/test")
val_path = os.path.abspath("experiment_small_data/val")


def one_hot_encode(img_path, get_encoded=False):
    if img_path.split(".")[0][-3:] == 'cat':
        if get_encoded:
            return 1
        """
        t = img_path.split(".")[0][-3:]
        tt = img_path.split(".")[0]
        ttt = img_path.split(".")"""
        return 'cat'
    else:
        if get_encoded:
            return 0
        return 'dog'


# glob finds all the pathnames matching a specified pattern according to the rules used by the Unix shell
train_image = [(plt.imread(img_path), one_hot_encode(img_path)) for img_path in glob.glob(f'{current_dir}/experiment_small_data/train/*.jpg')]
test_image = [(plt.imread(img_path), one_hot_encode(img_path)) for img_path in glob.glob(f'{current_dir}/experiment_small_data/test/*.jpg')]
val_image = [(plt.imread(img_path), one_hot_encode(img_path)) for img_path in glob.glob(f'{current_dir}/experiment_small_data/val/*.jpg')]

# bra sida att kolla på
# https://studymachinelearning.com/keras-imagedatagenerator-with-flow/

# one hot encode exempel, valde dock att lägga in det i en funktion som anropas i div funktioner
#labels = image_small_data
#labels = np.array([1 if label[:3] == 'cat' else 0 for label in labels])


def display_images_encoded(data, get_labels, title, nrows=2, ncols=5, figsize=(15, 8)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    # KLAR
    for i, ax in enumerate(axes.flatten()):
        if get_labels:
            ax.imshow(data[i][0], cmap='gray')
            ax.axis("off")
            ax.set_title(data[i][1], fontsize=12)
        else:
            ax.imshow(data[i], cmap='gray')
            ax.axis("off")

    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(wspace=0, hspace=.1, bottom=0)
    plt.show()

"""
display_images_encoded(val_image[:10], title="10 Val images - small data", get_labels=True)
display_images_encoded(test_image[:10], title="10 Test images - small data", get_labels=True)
display_images_encoded(train_image[:10], title="10 Train images - small data", get_labels=True)
"""


def count_plot(x, title):  # KLAR
    sns.countplot(x=x)
    plt.title(title)
    plt.show()


x_train = pd.DataFrame([img[1] for img in train_image])[0]
x_test = pd.DataFrame([img[1] for img in test_image])[0]
x_val = pd.DataFrame([img[1] for img in val_image])[0]
"""count_plot(x_train, "Train_images")
count_plot(x_test, "Test_images")
count_plot(x_val,  "Val_images")
"""


def scatter_plot(df_lst, title_lst, get_labels=False):
    # ej klar
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, df, title in zip(axes.flatten(), df_lst, title_lst):
        sns.scatterplot(data=df, ax=ax)
        ax.set_title(title)
    plt.show()


#img_val = pd.DataFrame([img[0] for img in val_image])[0]
#scatter_plot([pd.DataFrame([img[0] for img in val_image]), pd.DataFrame([img[0] for img in test_image]),
#              pd.DataFrame([img[0] for img in test_image])],
#             ["small data - val", "small data - test", "small data - train"])

# gamla :
# val_collect_encoded, test_collect_encoded, train_collect_encoded
# , test_image, train_image , "small data - test", "small data - train"


train_image = [(plt.imread(img_path), one_hot_encode(img_path)) for img_path in glob.glob(f'{current_dir}/experiment_small_data/train/*.jpg')]
test_image = [(plt.imread(img_path), one_hot_encode(img_path)) for img_path in glob.glob(f'{current_dir}/experiment_small_data/test/*.jpg')]
val_image = [(plt.imread(img_path), one_hot_encode(img_path)) for img_path in glob.glob(f'{current_dir}/experiment_small_data/val/*.jpg')]


def get_size_of_images(img_lst):
    return [img[0].shape for img in img_lst]


img_sizes = pd.DataFrame(get_size_of_images(train_image) + get_size_of_images(test_image) +
                         get_size_of_images(val_image), columns=["height", "width", "color"])

print("Min:", img_sizes.min(), "\nMax:", img_sizes.max())
""" In i rapporten
Min: 
height    37.0
width      3.0
color      3.0

Max: 
height    500.0
width     500.0
color       3.0
"""


def joint_plot(df_sizes):
    # KLAR
    sns.jointplot(data=df_sizes, x="height", y="width")     # testat kind="reg" hist, reg
    plt.show()
    ax1 = sns.countplot(data=df_sizes, x="height")
    ax1.set_title("Count Height")
    ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    plt.show()
    ax2 = sns.countplot(data=df_sizes, x="width")
    ax2.set_title("Count Width")
    ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(50))
    plt.show()


#joint_plot(img_sizes)

""" många som ligger 500x500 några få under 100x100, det ligger en ansamling vid 200x200, de flesta är över 160x160
bilder under 160x160 eller 200x200 bör slängas tror jag resized_val_images = [(cv2.resize(image[0], (100, 100)), image[1]) for image in data]
"""


def augment_data(data):

    #data_resized = [cv2.resize(data[i], (300, 300)) for i in range(len(data))]
    #resized_shape = [data_resized[i].shape for i in range(len(data_resized))]
    resized_val_images = [(cv2.resize(image[0], (300, 300)), image[1]) for image in data]  # testar men funkar inte
    print()
    return resized_val_images


train_resized = augment_data(train_image)
test_resized = augment_data(test_image)
val_resized = augment_data(val_image)


def display_resized(data, title="Resized", nrows=3, ncols=5, figsize=(15, 10)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(data[i], cmap="gray")     # ta bort cmap?
        ax.axis("off")
        fig.suptitle(title)

    fig.subplots_adjust(wspace=0, hspace=.1, bottom=0)
    plt.show()


random.shuffle(train_resized)
random.shuffle(test_resized)
random.shuffle(val_resized)

"""display_images_encoded(train_resized[:10], title="Train resized - small data", get_labels=True)
display_images_encoded(test_resized[:10], title="Test resized - small data", get_labels=True)
display_images_encoded(val_resized[:10], title="Val resized - small data", get_labels=True)"""


# Train|val|test split and Normalize data
scaled_X_train = np.array([image[0] for image in train_resized]).astype("float32")/255.0
scaled_X_test = np.array([image[0] for image in test_resized]).astype("float32")/255.0
scaled_X_val = np.array([image[0] for image in val_resized]).astype("float32")/255.0
print("scaled_X", scaled_X_train.shape, scaled_X_test.shape, scaled_X_val.shape)

# Train|val|test split
y_train = np.array([image[1] for image in train_resized])
y_test = np.array([image[1] for image in test_resized])
y_val = np.array([image[1] for image in val_resized])
print("scaled_y", y_train.shape, y_test.shape, y_val.shape)


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
# testa olika inställningar i taget!
train_image_generator = ImageDataGenerator(rotation_range=10, shear_range=.1, zoom_range=.1, horizontal_flip=False,
                                           height_shift_range=.2, width_shift_range=.2)

test_image_generator = ImageDataGenerator()
train_generator = train_image_generator.flow(scaled_X_train, y_train, batch_size=32)

val_generator = test_image_generator.flow(scaled_X_val, y_val, batch_size=32)

print(len(train_generator.next()))
sample_batch = train_generator.next()
print("sample_batch", len(sample_batch))
print("sample_batch[0]", sample_batch[0].shape)

display_images_encoded(sample_batch[0], title="augumented", get_labels=False)

# ej gjort något nedan

# CNN model
def CNN_model(learning_rate=.001, drop_rate=.5, kernels=[32, 32]):
    adam = Adam(learning_rate=learning_rate)

    model = Sequential(name="CNN_model")

    # the convolutional layers
    for number_kernel in kernels:
        conv_layer = Conv2D(number_kernel, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal",
                            input_shape=scaled_X_train.shape[1:])

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
steps_per_epoch = int(len(scaled_X_val) / 32)
validation_steps = int(len(scaled_X_val) / 32)

print(steps_per_epoch, validation_steps)

early_stopper = EarlyStopping(monitor="val_acc", mode="max", patience=5, restore_best_weights=True)

model.fit(val_generator, steps_per_epoch=steps_per_epoch, epochs=100, callbacks=[early_stopper],
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

