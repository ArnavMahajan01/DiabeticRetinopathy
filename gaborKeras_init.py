#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:05:06 2021

@author: gitanshwadhwa
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras.backend as K
from theano import shared

train_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('./sem-long/DiabeticRetinopathy/preprocessed_dataset',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')



test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('./sem-long/DiabeticRetinopathy/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


"""
def custom_gabor(dtype=None):
    ksize = 20
    sigma = 10
    theta = 0
    lamda = 2
    gamma = 0.5
    phi = 0
    kernels = []
    for theta in range(4):
        theta = theta/4 * np.pi
        for sigma in (1, 3, 5):
            for lamda in np.arange(0, np.pi, np.pi/4):
                for gamma in (0.05, 0.5):
                    ksize = 15
                    phi = 0.8
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                    kernels.append(kernel)
    np_tot = shared(np.array(kernels))
    return K.variable(np_tot, dtype = dtype)
             """
def custom_gabor(shape, dtype=None):
    total_ker = []
    for i in range(shape[3]):
        kernels = []
        for j in range(shape[2]):
            kernels.append(
            cv2.getGaborKernel(ksize=(shape[0], shape[1]), sigma=1, 
            theta=1, lambd=0.5, gamma=0.3, psi=(3.14) * 0.5, 
            ktype=cv2.CV_32F))
        total_ker.append(kernels)
    np_tot = shared(np.array(total_ker))
    return K.variable(np_tot, dtype=dtype)
            
cnn = tf.keras.models.Sequential()
"""
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,kernel_initializer=custom_gabor, activation='relu', input_shape=[64, 64, 1]))  
"""
cnn.add(tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer=custom_gabor,
                                      input_shape=(1, 64, 64)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())  


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))   

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])  

cnn.fit(x = training_set, validation_data = test_set, epochs = 5)    


from keras.preprocessing import image
test_image = image.load_img('./sem-long/DiabeticRetinopathy/test/xyz/38.tif', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'yes'
else:
  prediction = 'no'  

   
print(prediction)


"""

def custom_gabor(shape, dtype=None):
    total_ker = []
    for i in xrange(shape[3]):
        kernels = []
        for j in xrange(shape[2]):
            kernels.append(
            cv2.getGaborKernel(ksize=(shape[0], shape[1]), sigma=1, 
            theta=1, lambd=0.5, gamma=0.3, psi=(3.14) * 0.5, 
            ktype=CV_64F))
        total_ker.append(kernels)
    np_tot = shared(np.array(total_ker))
    return K.variable(np_tot, dtype=dtype)
 

def build_model():
    model = Sequential()
    # Layer 1
    model.add(Conv2D(32, (3, 3), kernel_initializer=custom_gabor,
                                      input_shape=(nb_channel, img_rows, img_cols)))
    model.add(Activation('relu'))
    # Layer 2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), kernel_initializer=custom_gabor))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    model.add(Conv2D(32, (3, 3), kernel_initializer=custom_gabor))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))
"""