import tensorflow as tf
import os


class Preprocessing:
    def __init__(self, main_directory = 'data', reduce = 255):
        self.main_directory = main_directory
        self.reduce = reduce

    def imagePreprocessing(self):
        db = tf.keras.utils.image_dataset_from_directory(os.path.join('libraries', self.main_directory))
        image_pixel = db.map(lambda x, y : (x/self.reduce, y))
        #scaled_iterator = image_pixel.as_numpy_iterator()
        #batch2 = scaled_iterator.next()
        
        train_size = int(len(image_pixel)*.7)
        val_size = int(len(image_pixel)*.2)+1
        test_size = int(len(image_pixel)*.1)

        train = image_pixel.take(train_size)
        val = image_pixel.skip(train_size).take(val_size)
        test = image_pixel.skip(train_size+val_size).take(test_size)
        #print(train_size, val_size, test_size)

        return train, val, test
        


