import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from scipy.misc import imresize
import numpy as np

from utils import get_image_paths

image_paths = get_image_paths()
img = load_img(image_paths[0], grayscale=True)
img_array = img_to_array(img)

#  reshaping to 32 * 120 images
print("Original image \t : {0}".format(img_array.shape))
img_array = imresize(img_array[:, :, 0], (32, 100)).astype("float32")
print("Reshaped image \t : {0}".format(img_array.shape))
img_array = img_array.reshape(32, 100, 1)
print("Final image \t : {0}".format(img_array.shape))

augmenting_datagen = ImageDataGenerator(
    rescale=None,
    rotation_range=None,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False,
    fill_mode='nearest',
)

plt.figure(figsize=(11, 5))
flow = augmenting_datagen.flow(img_array[np.newaxis, :, :, :])
for i, x_augmented in zip(range(15), flow):
    plt.subplot(3, 5, i + 1)
    plt.imshow(array_to_img(x_augmented[0]), cmap='gray')
    plt.axis('off')

# flow = augmenting_datagen.flow_from_directory(
#     data_folder_1, batch_size=1, target_size=(32, 120))

plt.figure(figsize=(11, 5))
for i, (X, y) in zip(range(15), flow):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X[0])
    plt.axis('off')
