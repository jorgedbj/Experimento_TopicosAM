from __future__ import print_function
import numpy as np
np.random.seed(1337)

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from sklearn.metrics import classification_report,confusion_matrix

from scipy import misc
import matplotlib.pyplot as plt
import webbrowser
import urllib
import hashlib

import os.path
import time

total = 120

batch_size = 6
nb_classes = 30
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 54, 54
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# Alocacao das matrizes do conjunto de treinamento e teste
X_train = np.zeros(total * img_rows * img_cols, dtype=int)
X_train = X_train.reshape(total, img_rows, img_rows)

X_test = np.zeros(30 * img_rows * img_cols, dtype=int)
X_test = X_test.reshape(30, img_rows, img_rows)

y_train = np.arange(30)
for i in range (1, 4):
    y_train = np.concatenate((y_train, np.arange(30)))

y_test = np.arange(30)

# Importacao das imagens do conjunto de treinamento
for i in range(0, 120):
    X_train[i] = misc.imread("/Users/JorgeJunior/Downloads/Dados-3/treino/" + str(i+1) + ".png", 1)

# Importacao das imagens do conjunto de teste
for i in range(0, 30):
    X_test[i] = misc.imread("/Users/JorgeJunior/Downloads/Dados-3/teste/" + str(i+1) + ".png", 1)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Conversao dos vetores dos labels de saida para o formato: one-hot vector
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Construcao da arquitetura da Rede Neural Convolucional (CNN)
model = Sequential()
model.add(Conv2D(nb_filters, kernel_size=(nb_conv,nb_conv),
                        activation='tanh',
                        input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Conv2D(nb_filters, (nb_conv,nb_conv), activation='tanh'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Conv2D(nb_filters, (nb_conv,nb_conv), activation='tanh'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Nadam(),
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

# Validacao da CNN
score = model.evaluate(X_test, Y_test, verbose=1)
classes = model.predict_classes(X_test)
proba = model.predict_proba(X_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#target_names = ['config0', 'config1', 'config2', 'config3', 'config4', 'config5', 'config6', 'config7', 'config8', 'config9', 'config10',
#                'config11', 'config12', 'config13', 'config14', 'config15', 'config16', 'config17', 'config18', 'config19', 'config20', 'config21',
#                'config22', 'config23', 'config24', 'config25', 'config26', 'config27', 'config28', 'config29']
#print(classification_report(np.argmax(Y_test,axis=1), classes,target_names=target_names))
#print(confusion_matrix(np.argmax(Y_test,axis=1), classes))