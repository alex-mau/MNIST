from __future__ import print_function
import numpy as np
np.random.seed(1)

from keras.utils import np_utils
import os
import struct

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join('/Users/alex/Documents/Python/mnist', '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join('/Users/alex/Documents/Python/mnist', '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

X_train, y_train = load_mnist('../data', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('../data', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

import theano
theano.config.floatX = 'float32'
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)
y_train_ohe = np_utils.to_categorical(y_train)
y_test_ohe = np_utils.to_categorical(y_test)
X_train=X_train/256
X_test=X_test/256

from keras.models import Sequential
from  keras import callbacks
from keras.optimizers import SGD
from keras.layers import Dense, Dropout,Flatten,Reshape
from keras.layers import Convolution2D, MaxPooling2D
model=Sequential()
model.add(Reshape((28,28,1),input_shape=(784,)))
model.add(Convolution2D(16,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(8,4,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(125,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10,activation='softmax'))

sgd = SGD(lr=0.001, momentum=.9)

tensorboard=callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train,
          y_train_ohe,
          epochs=50,
          batch_size=300,callbacks=[tensorboard],
          verbose=1,validation_data=[X_test,y_test_ohe]
          )

'''pred=model.predict_classes(X_test,verbose=0)
acc=np.sum(y_test==pred,axis=0)/X_test.shape[0]'''




