import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
from scipy.misc import imresize


def get_image_paths():
    image_paths = []
    for root, dirs, files in os.walk('mjsynth/'):
        for file in files:
            if 'DS_Store' in file:
                continue
            image_paths.append(os.path.join(root, file))
    return image_paths


def word_from_image_path(image_path):
    try:
        out = image_path.split("_")[1]
    except:
        out = None
    return out


def preprocess_image(img_array):
    return imresize(img_array[:, :, 0], (32, 100)).astype("float32")


def print_im(im):
    plt.imshow(array_to_img(im.reshape((im.shape[0], im.shape[1], 1))), cmap='gray')
