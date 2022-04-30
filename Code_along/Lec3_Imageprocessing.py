"""
Code along - Image processing
Paths
We will later read and write image data to different directories, so it is important to understand how paths work.
Note that there is a difference in how to work with paths in Jupyter notebook in contrast with
Python script when we work in Visual studio code.

- in Jupyter notebook the path is relative to where the file is
- in Python script when we click the play button it is relative to the working directory in the terminal
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# in Pyton script i Vscode or PyCharm
os.chdir(os.path.dirname(__file__))


# in Python script this path gives you where you are in the terminal
current_dir = os.path.abspath("")
print(f"{current_dir}")

# list all files in current dir
files_current_dir = os.listdir(current_dir)
print(f"{files_current_dir}")

data_dir = os.path.abspath("../data")
print(f"{data_dir}")

img = plt.imread(f"{data_dir}/Homer_Simpson_2006.png")
print(f"Original shape {img.shape}")
plt.show()
# want to make it smallers
resize_factor = .5
new_size = (int(img.shape[1]*resize_factor), int(img.shape[0]*resize_factor))

img = cv2.resize(img, new_size)
plt.imshow(img)
plt.axis("off")
print(img.shape)


# Image kernels
edge_filter = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])

blur_filter = np.array([[.0625, .125, .0625],
                        [.125, .25, .125],
                        [.0625, .125, .0625]])

# adjust the values of the filter and see different effects
outline_filter = np.array([[-1, -1, -1],
                           [-1, 17, -1],
                           [-1, -1, -1]])

filters = [edge_filter, blur_filter, outline_filter]
filter_names = ["Edge", "Blur", "Outline"]

processed_images_path = f"{data_dir}/processed_images"

# create the folder processed_images inside data folder
try:
    os.mkdir(processed_images_path)
except FileExistsError as err:
    print("Already created folder")

fig, axes = plt.subplots(1,len(filters))


for ax, filter, filter_name in zip(axes, filters, filter_names):
    filtered_img = cv2.filter2D(img, -1, kernel=filter)
    ax.imshow(filtered_img)
    ax.axis("off")
    ax.set(title = f"{filter_name} kernel")  # ej rescaletat
plt.show()

fig.savefig(f"{processed_images_path}/Filtered_images.png")

source_path = f"{processed_images_path}/Filtered_images.png"
target_path = f"{processed_images_path}/Filtered_images_copied.png"
shutil.copyfile(source_path, target_path)
