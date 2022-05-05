# Transfer learning - CNN

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, Xception
(train, test), info = tfds.load("tf_flowers", split=["train[:80%]", "train[80%:]"], as_supervised=True,
                                with_info=True)
# lägg märke till : i split "train[]" kan splita i train, test val också, kolla på tensorflow

print(info)

# labels från data info eller från tensorflows hemsida, de har massa statistik för vare dataset de har där
label_names = ["dandelion", "daisy", "tulips", "sunflowers", "roses"]

print(label_names)

# tensorflow grejer här, inte fetchat datasetet ännu
print(train)

fig, axes = plt.subplots(2, 4, figsize=(12, 4))

for i, img_sample in enumerate(train.take(8)):  # take plockar ett visst antal- som står inom ()
    #print(f"img shape: {img_sample[0].shape}, label: {img_sample[1]}")
    ax = axes.flatten()[i]
    ax.imshow(img_sample[0])
    ax.set_title(f"{img_sample[1]}: {label_names[img_sample[1]]}")
    ax.axis("off")
    #plt.show()

"""
Preprocessing
    need to reshape the images
    scale them
    create batches
    optimize performance utilizing CPU and GPU with prefetching a batch
"""


def preprocess_images(img, label, img_shape=(120, 120)):
    img = tf.image.resize(img, img_shape) / 255.0
    return img, label


input_shape = 120, 120, 3  # det är 3 färgkanaler
batch_size = 32     # 32 eller 64 eller 128 verkar gå på bitstorlek ev på datorn Kolla upp

# prefetch always get one batch of data ready
# GPu work on backpropagation and forward propagation while CPU works on preprocessing a batch
# buffer size ska vara mindre än totala antalet i datasetet - se mer på tensorflow
train_batch = train.shuffle(buffer_size=600).map(preprocess_images).batch(batch_size).prefetch(1)
test_batch = test.map(preprocess_images).batch(batch_size).prefetch(1)

print(train_batch)
print(test_batch)

"""
Transfer learning
There a few approaches to transfer learning and the approach we're using in this lecture note is:
Load a model which has been pretrained on a large dataset e.g. imagenet.
Remove the top, i.e. the classifier part and MLP part
Freeze the layers weights of the pretrained network. Reason for this is to use the pretrained weights to extract 
feature maps from new images which it hasn't seen before.
Add an MLP part and classifier part. Train the data on the last part.
Idea behind this is that we can reuse a network that is trained on large dataset as many kernels such as edge detectors,
simple shape detectors can be used on other datasets as well. Also this is cheap as it is more data-expensive 
and computationally expensive to train a large network from scratch.

However some deeper layers extract more and more complicated structures that may be too specific for another dataset. 
Then another approach is to use the shallower layers for feature extraction and train the new dataset on the deeper 
layers to fine-tune the network. Then add an MLP part as in above.
"""
# Transfer learning
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(input_shape))
base_model.summary()

# ett annat sätt än add används här, det finns flera sätt
model = Sequential([base_model, Flatten(), Dropout(0.5), Dense(256, activation="relu", kernel_initializer="he_normal"),
                    Dropout(0.5), Dense(5, activation="softmax")], name="cool_model")   # dense(5 - fem labels!!!

# fryser layers vikter och ev bias
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
model.summary()

model.fit(train_batch, epochs=20, validation_data=test_batch)

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[["loss", "val_loss"]].plot()
metrics[["acc", "val_acc"]].plot()
plt.show()


base_model2 = Xception(weights="imagenet", include_top=False, input_shape=(input_shape))
base_model2.summary()

# ett annat sätt än add används här, det finns flera sätt
model2 = Sequential([base_model2, GlobalAveragePooling2D(), Dropout(0.5), Dense(256, activation="relu", kernel_initializer="he_normal"),
                    Dropout(0.5), Dense(256, activation="softmax")], name="cool_model")   # dense(256 - 256 labels!!!

# fryser layers vikter och ev bias
for layer in base_model2.layers:
    layer.trainable = False

model2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
model2.summary()

model2.fit(train_batch, epochs=20, validation_data=test_batch)

metrics2 = pd.DataFrame(model2.history.history)
metrics2.head()

metrics2[["loss", "val_loss"]].plot()
metrics2[["acc", "val_acc"]].plot()
plt.show()
