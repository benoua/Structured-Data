import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
from keras.preprocessing.text import Tokenizer
from scipy.misc import imresize
import warnings
import numpy as np

from alphanum_symbols import char2ix

N_CHARS = 37
SEQUENCE_LENGTH = 23
IMAGE_DIMENSIONS = (32, 100)
# IMAGE_DIMENSIONS = (10, 30)


def get_image_paths(base_dir):
    image_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if 'DS_Store' in file:
                continue
            if file.endswith('.txt'):
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
    return imresize(img_array[:, :, 0], IMAGE_DIMENSIONS).astype("float32")


def print_im(im):
    plt.imshow(array_to_img(im.reshape((im.shape[0], im.shape[1], 1))), cmap='gray')


class TextTransform(object):
    def __init__(self):
        self.tokenizer = Tokenizer(nb_words=N_CHARS, char_level=True)
        self.tokenizer.fit_on_texts([''.join(char2ix)])
        self.inv_vocabulary = {v: k for k, v in self.tokenizer.word_index.items()}

    def transform(self, word):
        return self.tokenizer.texts_to_matrix(word.ljust(SEQUENCE_LENGTH).lower())

    def word_from_matrix(self, m):
        try:
            return ''.join([self.inv_vocabulary[l] for l in m.argmax(axis=1)])
        except KeyError:
            warnings.warn("missing char")
            return ''

    def make_batch_labels(self, image_paths):
        names = [word_from_image_path(filename).lower() for filename in image_paths]
        return np.array([self.transform(name) for name in names])
