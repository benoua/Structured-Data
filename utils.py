import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
from keras.preprocessing.text import Tokenizer
from scipy.misc import imresize

from alphanum_symbols import char2ix

N_CHARS = 37
SEQUENCE_LENGTH = 23

IMAGE_DIMENSIONS = (32, 100)

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
    return imresize(img_array[:, :, 0], IMAGE_DIMENSIONS).astype("float32")


def print_im(im):
    plt.imshow(array_to_img(im.reshape((im.shape[0], im.shape[1], 1))), cmap='gray')


class TextTransform(object):
    def __init__(self):
        self.tokenizer = Tokenizer(nb_words=SEQUENCE_LENGTH, char_level=True)
        self.tokenizer.fit_on_texts([''.join(char2ix.keys())])
        self.inv_vocabulary = {v: k for k, v in self.tokenizer.word_index.items()}

    def transform(self, word):
        return self.tokenizer.texts_to_matrix(word.ljust(N_CHARS).lower())

    def word_from_matrix(self, m):
        ''.join([self.inv_vocabulary[l] for l in m.argmax(axis=1)])
