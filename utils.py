import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
from keras.preprocessing.text import Tokenizer
from scipy.misc import imresize
import warnings
import numpy as np
import pickle
from scipy.sparse import issparse

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
        out = image_path.split("_")[-2]
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


def my_send_mail(subject, body):
    user = 'api.maxbellec'
    pwd = 'apiApisendmail'
    send_email(user, pwd, user, subject, body)


def send_email(user, pwd, recipient, subject, body):
    import smtplib

    gmail_user = user
    gmail_pwd = pwd
    FROM = user
    TO = recipient if type(recipient) is list else [recipient]

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), subject, body)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print('successfully sent the mail')
    except:
        print("failed to send mail")


class NgramTransform(object):
    def __init__(self):
        """
        Class that convert words to 10k most used N-grams in synth90k.
        """
        self.cv = pickle.load(open("cv2.pkl", "rb"))
        self.X_tf = np.array(pickle.load(open("Xtf.pkl", "rb")))[0]
        self.vocabulary = self.cv.vocabulary_
        self.inv_vocabulary = {v: k for k, v in self.cv.vocabulary_.items()}

    def transform(self, word, sparse=True):
        """

        :param word: string (one word at a time)
        :param sparse: if True, return sparse matrix, otherwise, numpy array
        :return: vector with 1 on idx if presence of this N-gram
        """
        if sparse == True:
            return self.cv.transform([word.lower()])
        else:
            out = self.cv.transform([word.lower()]).toarray()
            return out > 0  # convert to bolean vector

    def ngram_from_matrix(self, m):
        """

        :param m: flat matrix
        :return: list of N-grams
        """
        if issparse(m):
            m = m.toarray()[0]
        idx_ngrams = np.where(m == 1)[0]
        try:
            return [self.inv_vocabulary[l] for l in idx_ngrams]
        except KeyError:
            warnings.warn("missing char")
            return ''

    def make_batch_labels(self, image_paths, sparse=True):
        """

        :param sparse: if True, return sparse matrix, else arrays
        :param image_paths: path of images
        :return: list of all N-grams for all images names.
        """
        names = [word_from_image_path(filename).lower() for filename in image_paths]
        return np.array([self.transform(name, sparse) for name in names])


def load_trained_CNN_weights(p_model, dense_layer=True):
    """

    :param p_model: trained parallel model to import
    :return: CNN model with weights of // model
    """
    from keras.models import load_model
    cnn = load_model(p_model)

    try:
        cnn_empty = load_model('model_pretrained_weights.keras')
    except KeyError:
        warnings.warn("model_pretrained_weights.keras not in folder")
    all_weights = cnn.layers[5].get_weights()


    saved_weights = {}
    for i in range(5):
        saved_weights['convo{}'.format(i)] = [all_weights[i * 2], all_weights[i * 2 + 1]]
    for i in range(3):
        saved_weights['Dense{}'.format(i + 1)] = [all_weights[10 + i * 2], all_weights[10 + i * 2 + 1]]

    if dense_layer == True:
        layers_names = ['convo0', 'convo1', 'convo2', 'convo3', 'convo4', 'Dense1', 'Dense2', 'Dense3', ]
    elif dense_layer == False:
        layers_names = ['convo0', 'convo1', 'convo2', 'convo3', 'convo4', ]

    # setting weights of each layers
    try:
        for layer_name in layers_names:
            layer = cnn_empty.get_layer(name=layer_name)
            layer.set_weights(saved_weights[layer_name])
    except KeyError:
        warnings.warn("wrong model")
    return cnn_empty

def plot_batch_images(x,y, name=None, predictions=None):
    plt.figure(figsize=(10,5))
    idxs = np.random.randint(0, x.shape[0], 20)



    if name=="benoit":
        with open('tt_new.pickle', 'rb') as f:
            tt = pickle.load(f)
    else :
        with open('/datadrive/tt_new.pickle', 'rb') as f:
            tt = pickle.load(f)

    for i,idx in enumerate(idxs):
        plt.subplot(4, 5, i + 1)
        plt.imshow(x[idx], cmap='gray')
        plt.axis('off')
        if predictions == None:
            plt.title(y[idx])
        else:
            plt.title(tt.word_from_matrix(y[idx]))

def base_cnn_in_keras():
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Reshape, Activation
    from keras.layers import Flatten

    convolutions = [64, 128, 256, 512, 512]
    kernels = [3, 3, 3, 3, 3]

    model = Sequential()
    input_shape = (None,) + IMAGE_DIMENSIONS + (1,)

    model.add(Conv2D(nb_filter=64,
                     nb_row=kernels[0],
                     nb_col=kernels[0],
                     activation='relu',
                     border_mode='same',
                     batch_input_shape=input_shape, name="convo" + str(0)))

    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

    for i, (kernel, convolution_size) in enumerate(zip(kernels[1:], convolutions[1:])):
        model.add(Conv2D(nb_filter=convolution_size,
                         nb_row=kernel,
                         nb_col=kernel,
                         activation='relu',
                         border_mode='same',
                         name="convo" + str(i + 1)))

        if i <= 1:
            model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same', ))

    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(4096, activation='relu', name="Dense1"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name="Dense2"))
    model.add(Dropout(0.5))

    return model

def raw2gray(im):
    """
        Convert a raw image to its grayscale image between 0 and 1.

        Arguments:
            - im : raw 2D image

        Returns:
            - im_gray : grayscale 2D image
    """
    # average all RGB channels
    im_gray = im.mean(axis=2)

    # normalize the image between 0 and 1
    im_gray = (im_gray - im_gray.min()) / (im_gray.max() - im_gray.min())

    return im_gray
