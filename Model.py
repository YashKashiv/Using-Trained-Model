import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import time
from tensorflow.keras.callbacks import TensorBoard

pickle_in = open("x.pickle", "rb")
x = pickle.load(pickle_in) # pickle files created on Day 2

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)


x = np.array(x)
y = np.array(y)

x = x/255.0

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            name = "{}-conv-{}-nodes-{}--dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            
            print(name)

            model = Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape = x.shape[1:])) # First Layer
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3))) # Second Layer
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))
            
            model.add(Dense(64))
            model.add(Activation("relu"))

            model.add(Dense(1)) # Output Layer
            model.add(Activation("sigmoid"))

            #tensorboard = TensorBoard(log_dir="logs/{}".format(name))
            #model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

            #model.fit(x, y, batch_size=32, epochs=3, validation_split=0.1, callbacks=[tensorboard])

model.save("64x3-CNN.model.keras")