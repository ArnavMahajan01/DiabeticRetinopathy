#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation, Conv2D, InputLayer
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import cv2
import keras.backend as K
from keras.engine.input_layer import Input

train_path = 'train'
valid_path = 'valid'
test_path = 'test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['dr','nodr'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['dr','nodr'], batch_size=10)

# def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
#     if type(ims[0] is np.ndarray):
#         ims = np.array(ims).astype(np.uint8)
#         if(ims.shape[-1] != 3):
#             ims = ims.transpose((0,2,3,1))
#     f = plt.figure(figsize=figsize)
#     cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
#     for i in range(len(ims)):
#         sp = f.add_subplot(rows, cols, i+1)
#         sp.axis('Off')
#         if titles is not None:
#             sp.set_title(titles[i], fontsize=16)
#         w = plt.waitforbuttonpress()
#         print(w)
#         plt.imshow(ims[i], interpolation=None if interp else 'none')

def gaborConvolutionalLayer2(shape, dtype=None):
    print(shape)
    total_ker = np.zeros(shape)
    # s4 = [0]*shape[3]
    # three = [s4]*shape[2]
    # five = [three]*shape[0]
    # total_ker = [five, five, five, five, five]
    c = 0
    for theta in range(16):
        theta = theta/16 * np.pi
        for lamda in np.arange(1, np.pi+1, np.pi/4):
            ksize = shape[0]
            phi = 0.8
            kernel = cv2.getGaborKernel((ksize, ksize), 3, theta, lamda, 0.05, phi, ktype=cv2.CV_64F)
            # for s in range(shape[2]):
            #     total_ker[:, :, s, c] = kernel
            for i in range(shape[0]):
                for j in range(shape[1]):
                    total_ker[i][j][0][c] = kernel[i][j]
                    total_ker[i][j][1][c] = kernel[i][j]
                    total_ker[i][j][2][c] = kernel[i][j]
            c = c+1
    total_ker = np.array(total_ker)
    assert total_ker.shape == shape
    vr = K.variable(total_ker, dtype=dtype)
    return vr


imgs , labels = next(train_batches)

convLayer = Conv2D(filters=64, kernel_size=(5, 5), kernel_initializer=gaborConvolutionalLayer2, input_shape=(228, 228, 3), activation="elu")

vgg16_model = keras.applications.vgg16.VGG16()
vgg19_model = keras.applications.vgg19.VGG19()

model1 = Sequential()
model1.add(convLayer)
model2 = Sequential()



# # print(model.layers[0].input_shape)
# # print(model.layers[0].output_shape)

# # print(vgg16_model.layers[1].input_shape)
# # print(vgg16_model.layers[1].output_shape)

# # print(vgg16_model.layers[2].input_shape)
# # print(vgg16_model.layers[2].output_shape)


for layer in vgg16_model.layers[2:-1]:
    model1.add(layer)

    
for layer in model1.layers:
    layer.trainable = False

model1.layers[0].trainable = True

model1.add(Dense(2, activation='softmax'))
model1.summary()


model1.compile(Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])

model1.fit(train_batches, steps_per_epoch=4,
validation_data=valid_batches, validation_steps=4, epochs=100, verbose=2)

model1.save("my_model1.h5")

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(228,228), classes=['dr','nodr'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(228,228), classes=['dr','nodr'], batch_size=10)

for layer in vgg19_model.layers[:-1]:
    model2.add(layer)

    
for layer in model2.layers:
    layer.trainable = False


model2.add(Dense(2, activation='softmax'))


model2.compile(Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(train_batches, steps_per_epoch=4,
validation_data=valid_batches, validation_steps=4, epochs=100, verbose=2)

model2.save("my_model2.h5")



