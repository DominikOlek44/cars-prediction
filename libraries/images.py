import tensorflow as tf
import os
import cv2
import imghdr


class Images:
    def __init__(self, main_directory = 'data', cars_dir= 'cars', without_cars = 'without_cars'):
        self.main_directory = main_directory
        self.cars_dir = cars_dir
        self.without_cars = without_cars

    def creatingDirs(self):
        root_path = 'libraries'
        os.mkdir(os.path.join(root_path, self.main_directory))
        sub_folders = [self.cars_dir, self.without_cars]
        try:
            for folder in sub_folders:
                os.mkdir(os.path.join(root_path, self.main_directory, folder))
        except OSError as e:
            print(e)


    def deletingWrongImages(self):
        # function deletes wrong image extension
        ext = ['jpeg', 'jpg', 'bmp', 'png']
        directory = self.main_directory

        for dir in os.listdir(directory):
            for image in os.listdir(os.path.join(directory, dir)):
                try:   
                    images = os.path.join(directory, dir, image)
                    img = cv2.imread(images)
                    type = imghdr.what(images)
                    if type not in ext:
                        print("Wrong image extension {}, have been removed".format(images))
                        os.remove(images)
                except Exception as e:
                    print("there is a problem with image {}".format(images))

    def limitGPU(self):
        # Setting limitation for GPU memory
        gpus = tf.config.list_physical_devices('GPU')

        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    pre = Preprocessing()
    pre.creatingDirs()
