import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

# How to process img into numpy using OpenCV


class Classifier:
    def __init__(self, labels, X, y, path):
        self.labels = labels
        self.X = X
        self.y = y
        self.path = path

        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []


    def ImageToArray(self, file):
        img_arr = cv2.imread(file)   # reads an image in the BGR format
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)   # BGR -> RGB
        return img_arr


    def ProcessImages(self):

        # labels= ['normal','covid','pneumonia']
        # X = []
        names = []  # not used, but just for seeing the file names
        # y = []

        for label in self.labels:
            # index cos doing all the files is too intensive
            for filename in os.listdir(self.path+label+"/images/")[:5]:
                # dont run this yet with all cos it will crash # divide by 255 to normalise the data
                file = str(f"{self.path}{label}/images/{filename}")
                arr = self.ImageToArray(file)
                # having issues with appending np array (it clears the array each time, so we conver to python list first and then make the whole thing a np array later
                arr = arr[::2, ::2].tolist() #convert to list so we can append and also shrink resolution to 150x150
                label_index = self.labels.index(label)
                self.X.append(arr)
                self.y = np.append(self.y, label_index)


    def ProcessArrays(self):
        self.X = np.array(self.X)
        self.X = self.X/255
        self.X = self.X.astype('float32')

        self.y = np.array(self.y, dtype='int8')
        # y = tf.one_hot(y, 3)
        # one hot encode outputs
        self.y = np_utils.to_categorical(self.y)

    def SplitDataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

if __name__ == '__main__':
    # Only executed if you start this script as the main script,
    # i.e. you enter 'python path/to/wumpus.py' in a terminal.
    # Assuming you saved the script in the directory 'path/to'
    # and named it 'wumpus.py'.

    # TODO: In the original game you can replay a dungeon (same positions of you and the threats)

    c1 = Classifier(['normal', 'covid', 'pneumonia'], [], [], "./dataset/")
    c1.ProcessImages()
    c1.ProcessArrays()
    c1.SplitDataset()
    