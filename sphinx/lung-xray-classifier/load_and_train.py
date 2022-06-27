import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
# from PIL import Image
import os
import time

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

import pickle

"""_summary_
"""
if tf.test.gpu_device_name():
	print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
	print("Please install GPU version of TF")

if tf.test.gpu_device_name():
	print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
	is_gpu = len(tf.config.list_physical_devices('GPU')) > 0
	# print(is_gpu)
else:
	print("Please install GPU version of TF")


class ModelGenerator:
	""""_summary_"""
	def __init__(self, labels, path):
		"""_summary_

		Args:
			labels (_type_): _description_
			path (_type_): _description_
		"""
		self.labels = labels
		print(labels)
		self.X = []
		self.y = []
		self.path = path

		self.X_train = []
		self.X_test = []
		self.y_train = []
		self.y_test = []
		self.model = Sequential()

		self.MODEL_NAME = ""

	def ImageToArray(self, file):
		"""_summary_

		Args:
			file (_type_): _description_

		Returns:
			_type_: _description_
		"""
		img_arr = cv2.imread(file)   # reads an image in the BGR format
		img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)   # BGR -> RGB
		return img_arr

	def ProcessImages(self):
		"""_summary_"""

		names = []  # not used, but just for seeing the file names

		for label in self.labels:
			# index cos doing all the files is too intensive

			for filename in os.listdir(self.path+label+"/images/")[:3615]: # only 3615 files available
				# divide by 255 to normalise the data
				file = str(f"{self.path}{label}/images/{filename}")
				arr = self.ImageToArray(file)
				# having issues with appending np array (it clears the array each time, so we conver to python list first and then make the whole thing a np array later
				arr = arr[::2, ::2].tolist() #convert to list so we can append and also shrink resolution to 150x150
				label_index = self.labels.index(label)
				self.X.append(arr)
				self.y = np.append(self.y, label_index)
			print(f"DONE: {label}")

	def ProcessArrays(self):
		"""_summary_"""
		self.X = np.array(self.X)
		self.X = self.X/255
		self.X = self.X.astype('float32')

		self.y = np.array(self.y, dtype='int8')
		# y = tf.one_hot(y, 3)
		# one hot encode outputs
		self.y = np_utils.to_categorical(self.y)

	def SplitDataset(self):
		"""_summary_
		"""
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

	def BuildModel(self):
		"""Add layers to the model
		"""
		# print(self.labels)
		self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 150, 3))) # shape = X.shape[1:]
		self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(MaxPooling2D((2, 2)))
		self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(MaxPooling2D((2, 2)))
		self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(MaxPooling2D((2, 2)))

		# example output part of the model
		self.model.add(Flatten())
		self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
		self.model.add(Dense(len(self.labels), activation='softmax')) # final layer dense 2 since we have 2 labels

		# compile model
		# opt = SGD(lr=0.001, momentum=0.9)
		self.model.compile(
					  loss='binary_crossentropy',  # USE SPARSE if WE ARE USING THE ACTUAL NUMBERS E.G 1,2,3 BUT WE ALR 1HOT ENCODED THEM SO ITS G
					  metrics=['accuracy'],
					  optimizer='adam'
					 )
		print(self.model.summary())


	def TrainModel(self):
		"""_summary_
		"""
		# print(self.X_train.shape)
		# print(self.y_train.shape)

		history = self.model.fit(
			self.X_train,
			self.y_train,
			epochs=20,
			batch_size=32,
			validation_data=(self.X_test,self.y_test)
		)

	def EvaluateModel(self):
		"""_summary_
		"""
		evaluate = self.model.evaluate(self.X_test, self.y_test, verbose=0)
		print(evaluate)

	def SaveModel(self):
		"""Save the model so that it can be accessed later without having to recreate it
		"""
#         NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"

		self.model.save(f"./6-conv-128-nodes-2-dense-{int(time.time())}.model")

		f = open('labels.pickle', "wb")
		f.write(pickle.dumps(self.labels))
		f.close()

if __name__ == '__main__':
    """main function which is responsible for the overall creation of the model
    """
    gen = ModelGenerator(['normal','covid'], "./dataset/")
    gen.ProcessImages()
    gen.ProcessArrays()
    gen.SplitDataset
    gen.BuildModel()
    gen.TrainModel()
    gen.EvaluateModel()
    gen.SaveModel()
    
	# Only executed if you start this script as the main script,
	# i.e. you enter 'python path/to/main.py' in a terminal.
	# Assuming you saved the script in the directory 'path/to'
	# and named it 'main.py'.
   
	# gen = ModelGenerator(['normal', 'covid'],"./dataset/")
	# gen.ProcessImages()
	# gen.ProcessArrays()
	# gen.SplitDataset()
	# gen.BuildModel()
	# gen.TrainModel()
	# gen.EvaluateModel()
	# gen.SaveModel()