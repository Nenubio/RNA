# -*- coding: UTF-8 -*-
# Author:Chenyb
import numpy as np
import  pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Convolution1D, MaxPooling1D, Flatten
from keras.optimizers import Adam

np.random.seed(1337)

# input data
data1 = open('data_inputx.txt')
data2 = open('data_inputy.txt')
x_train = np.loadtxt("data_inputx.txt")
y_train = np.loadtxt("data_inputy.txt")
x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)

# data pre-processing

X_train = x_train
X_test = X_train[0: 1]
y_train = y_train
y_test = y_train[0: 1]

data1.close()
data2.close()

# Build CNN VGG
model = Sequential()

# Conv block 1 output shape (64，84)
model.add(Convolution1D(
 input_shape=(1, 84),   # num of sequence
 filters=64,
 kernel_size=3,
 strides=1,
 padding='same',
 activation='relu',
 name='conv1_1'
))
model.add(Convolution1D(64, 3, strides=1, padding='same', activation='relu', name='conv1_2'))
# Pooling block 1 (max pooling) output shape (64, 42)
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same', name='pool1'))

# Conv block 2 output shape (128, 42)
model.add(Convolution1D(128, 3, strides=1, padding='same', activation='relu', name='conv2_1'))
model.add(Convolution1D(128, 3, strides=1, padding='same', activation='relu', name='conv2_2'))
# Pooling block 2 (max pooling) output shape (128, 21)
model.add(MaxPooling1D(2, 2, 'same', name='pool2'))

# Conv block 3 output shape (256, 1, 21)
model.add(Convolution1D(256, 3, strides=1, padding='same', activation='relu', name='conv3_1'))
model.add(Convolution1D(256, 3, strides=1, padding='same', activation='relu', name='conv3_2'))
model.add(Convolution1D(256, 3, strides=1, padding='same', activation='relu', name='conv3_3'))
# Pooling block 3 (max pooling) output shape (256, 1, 11)
model.add(MaxPooling1D(2, 2, 'same', name='pool3'))

# Conv block 4 output shape (512, 1, 11)
model.add(Convolution1D(512, 3, strides=1, padding='same', activation='relu', name='conv4_1'))
model.add(Convolution1D(512, 3, strides=1, padding='same', activation='relu', name='conv4_2'))
model.add(Convolution1D(512, 3, strides=1, padding='same', activation='relu', name='conv4_3'))
# Pooling block 4 (max pooling) output shape (512, 1, 6)
model.add(MaxPooling1D(2, 2, 'same', name='pool4'))

# Conv block 5 output shape (512, 1, 6)
model.add(Convolution1D(512, 3, strides=1, padding='same', activation='relu', name='conv5_1'))
model.add(Convolution1D(512, 3, strides=1, padding='same', activation='relu', name='conv5_2'))
model.add(Convolution1D(512, 3, strides=1, padding='same', activation='relu', name='conv5_3'))
# Pooling block 5 (max pooling) output shape (512, 1, 3)
model.add(MaxPooling1D(2, 2, 'same', name='pool5'))

# Fully connected layer 1 input shape (512 * 1 * 3) = (1536), output shape (1536)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(64, activation='relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10, activation='softmax'))

# Save model
model.save('CNN_model.h5')

# Define optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
loss='categorical_crossentropy',
metrics=['accuracy'])
print('Training ------------')

# Another way to train the model
model.fit(X_train, y_train, epochs=100, batch_size=64,)
model.fit()
print('\nTesting ------------')

# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
