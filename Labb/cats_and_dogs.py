import seaborn as sns
import pandas as pd
import numpy as np
import os, random, shutil, glob
import cv2
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Flatten, Dense, InputLayer, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import keras_tuner as kt
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def display_images(data, path, nrows=2, ncols=5, figsize=(15, 8)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for pict, ax in zip(data, axes.flatten()):
        #img = plt.imread(os.path.join(path, pict))
        img = plt.imread(f"{path}/{pict}")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(pict, fontsize=12)

    fig.subplots_adjust(wspace=0, hspace=.1, bottom=0)
    plt.show()


#display_images(ten_train_images, original_data_train_dir)
#display_images(ten_test_images, original_data_test_dir)


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
folder_name_structure = ('experiment_small_dataset', 'experiment_tiny_dataset')
sub_folder_structure = ('test', 'train', 'val')
try:
    mkdir(f'{current_directory}/original_dataset')
    for folder_name in folder_name_structure:
        root_folder_name = f'{current_directory}/{folder_name}'
        mkdir(root_folder_name)
        for sub_folder_name in sub_folder_structure:
            mkdir(f'{root_folder_name}/{sub_folder_name}')
except:
    pass
"""

"""
d) Nu ska du göra train|val|test split med följande splits:
experiment_small
"""
# random.sample - Used for random sampling without replacement.
small_data = random.sample(os.listdir(original_data_train_dir), k=2500)
random.shuffle(small_data)

train = small_data[0:1600]
test = small_data[1600: 2100]
val = small_data[2100:]
#print(len(train), len(val), len(test))

train_path = os.path.abspath("experiment_small_data/train")
test_path = os.path.abspath("experiment_small_data/test")
val_path = os.path.abspath("experiment_small_data/val")


def save_data_to_folder(test_path, file_lst, original_dir, new_dir):
    if len(os.listdir(test_path)) != 0:
        files = glob.glob(new_dir)
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
            continue
    if len(os.listdir(test_path)) == 0:
        for filename in file_lst:
            if filename in os.listdir(original_dir):
                #print(filename)
                plt.savefig(os.path.join(test_path, filename))


save_data_to_folder(train_path, train, original_data_train_dir, "experiment_small_data/train/*.jpg")
save_data_to_folder(test_path, test, original_data_train_dir, "experiment_small_data/test/*.jpg")
save_data_to_folder(val_path, val, original_data_train_dir, "experiment_small_data/val/*.jpg")


"""
e) Läs in dataseten från experiment_small, experiment_tiny och plocka ut labelsvektorer, som ska vara
one-hot encoded med 0 och 1.
plotta några bilder med deras respektive labels och kontrollera att det är korrekt.
skapa lämplig plot för att kontrollera att dataseten är balanserade
skapa lämplig plot för att kontrollera att dataseten är slumpade (dvs inte ex [0, 0, ... 0, 1, 1, ..., 1]).
"""

labels = ["cat", "dog"]


def one_hot_encoder(df_to_convert, sub_lst):
    """
    :param df_to_convert: DataFrame
    :param sub_lst: list
    :return: DataFrame
    """

    # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features
    X = df_to_convert[sub_lst]  # drop='first' but if more choices than 2 'first' needed to be removed
    drop_enc = OneHotEncoder(handle_unknown='ignore').fit(X)
    encoded = drop_enc.transform(X).toarray()
    print(drop_enc.transform(X).toarray())
    print(drop_enc.get_feature_names_out())
    return encoded


encode = one_hot_encoder(val_path, ["cat.*.jpg", "dog.*.jpg"])
