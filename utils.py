import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
from keras.preprocessing.text import Tokenizer
from scipy.misc import imresize
import warnings
import numpy as np
import pickle

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


from scipy.sparse import issparse


from scipy.sparse import issparse

class NgramTransform(object):
    def __init__(self):
        """
        Class that convert words to 10k most used N-grams in synth90k.
        """
        self.cv = pickle.load(open("cv.pkl", "rb" ) )
        self.vocabulary = self.cv.vocabulary_
        self.inv_vocabulary = {v: k for k, v in self.cv.vocabulary_.items()}

    def transform(self, word, sparse= True):
        """

        :param word: string (one word at a time)
        :param sparse: if True, return sparse matrix, otherwise, numpy array
        :return: matrix with idx of Ngrams
        """
        if sparse == True:
            return self.cv.transform([word.lower()])
        else :
            return self.cv.transform([word.lower()]).toarray()


    def ngram_from_matrix(self, m):
        """

        :param m: flat matrix
        :return: list of N-grams
        """
        if issparse(m):
            m = m.toarray()[0]
        idx_ngrams = np.where(m==1)[0]
        try:
            return [self.inv_vocabulary[l] for l in idx_ngrams]
        except KeyError:
            warnings.warn("missing char")
            return ''

    def make_batch_labels(self, image_paths):
        """

        :param image_paths: path of images
        :return: list of all N-grams for all images names.
        """
        names = [word_from_image_path(filename).lower() for filename in image_paths]
        return np.array([self.transform(name) for name in names])