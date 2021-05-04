import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import keras
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam


def gaborConvolutionalLayer2(shape, dtype=None):
    print(shape)
    total_ker = np.zeros(shape)
    c = 0
    for theta in range(16):
        theta = theta/16 * np.pi
        for lamda in np.arange(1, np.pi+1, np.pi/4):
            ksize = shape[0]
            phi = 0.8
            kernel = cv2.getGaborKernel((ksize, ksize), 3, theta, lamda, 0.05, phi, ktype=cv2.CV_64F)
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

test_path= 'test'
train_path = 'train'
valid_path = 'valid'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(228,228), classes=['dr','nodr'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(228,228), classes=['dr','nodr'], batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(228,228), classes=["dr", "nodr"], batch_size=10)


custom = {"gaborConvolutionalLayer2":gaborConvolutionalLayer2}

model1 = keras.models.load_model("my_model1.h5", custom_objects=custom)
model2 = keras.models.load_model("my_model2.h5")

model1.summary()
model2.summary()

j = 1
for layer in model1.layers:
    layer._name = "layer" + str(j)
    j = j+1

j = 1
for layer in model2.layers:
    layer._name = "l" + str(j)
    j = j+1

model1._name = "model1"
model2._name = "model2"


model1.summary()
model2.summary()

inputs = keras.layers.Input(shape=(228, 228, 3))

combined = keras.layers.Concatenate()([model1(inputs), model2(inputs)])
outputs = keras.layers.Dense(2)(combined)

model3 = keras.models.Model(inputs, outputs)
model1.trainable = False
model2.trainable = False

model3.compile(Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])

model3.fit(train_batches, steps_per_epoch=4,
validation_data=valid_batches, validation_steps=4, epochs=100, verbose=2)

model3.save("my_model3.h5")

# predictions_dr = model2.predict(test_batches)

# sum = 0
# for i in predictions_dr[0:80]:
#     if i[0] > i[1]:
#         sum = sum+1

# for i in predictions_dr[79:]:
#     if i[1] > i[0]:
#         sum = sum+1

# print(sum/160)
