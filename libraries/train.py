import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from preprocessing import *

class Train:
    def __init__(self, resolution = (256,256,3), optimizer = 'Adam'):
        self.resolution = resolution
        self.optimizer = optimizer

    def model(self):
        pre = Preprocessing()
        train, val, test = pre.imagePreprocessing()

        model = Sequential()

        model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape= self.resolution ))
        model.add(MaxPooling2D())

        model.add(Conv2D(32, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())

        model.add(Conv2D(16, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(self.optimizer, loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
        model.summary()

        logdir = 'logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

        precision = Precision()
        recall = Recall()
        accuracy = BinaryAccuracy()

        for b in test.as_numpy_iterator():
            x, y = b
            score = model.predict(x)
            precision.update_state(y, score)
            recall.update_state(y, score)
            accuracy.update_state(y, score)

        print(f'Precision:{precision.result().numpy()}, Recall:{recall.result().numpy()}, Accuracy:{accuracy.result().numpy()}')
