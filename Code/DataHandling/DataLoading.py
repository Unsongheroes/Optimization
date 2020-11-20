#Used to load the idx3_ubyte data format
import idx2numpy
#Used as the data object for images and labes
import numpy as np
#used to split data
import random
import math
from datetime import datetime

class Datasets:
    """ def load_mnist(base_path, folder_path): """
    def load_mnist(self, base_path, folder_path):
        #Load training data
        mnist_image_train_name = 'train-images'
        mnist_label_train_name = 'train-labels'
        data_type = '.idx3-ubyte'
        full_path = base_path + folder_path + mnist_image_train_name + data_type
        images_train_mnist = idx2numpy.convert_from_file(full_path)
        data_type = '.idx1-ubyte'
        full_path = base_path + folder_path + mnist_label_train_name + data_type
        labels_train_mnist = idx2numpy.convert_from_file(full_path)
        #Load test data
        mnist_image_test_name = 't10k-images'
        mnist_label_test_name = 't10k-labels'
        data_type = '.idx3-ubyte'
        full_path = base_path + folder_path + mnist_image_test_name + data_type
        images_test_mnist = idx2numpy.convert_from_file(full_path)
        data_type = '.idx1-ubyte'
        full_path = base_path + folder_path + mnist_label_test_name + data_type
        labels_test_mnist = idx2numpy.convert_from_file(full_path)
        #Return data
        return images_train_mnist, labels_train_mnist, images_test_mnist, labels_test_mnist

    """ def load_orl(base_path, folder_path): """
    def load_orl(self, base_path, folder_path):
        #Load data
        orl_image_name = 'orl_data'
        orl_lable_name = 'orl_lbls'
        data_type = '.txt'
        full_path = base_path + folder_path + orl_image_name + data_type
        images = np.loadtxt(full_path)
        images = images.reshape(30, 40, 400)
        images_orl = images.transpose()
        data_type = '.txt'
        full_path = base_path + folder_path + orl_lable_name + data_type
        labels_orl = np.loadtxt(full_path)
        #Return data
        return images_orl, labels_orl