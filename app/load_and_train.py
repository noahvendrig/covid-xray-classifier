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

	def CheckGPU():
		"""Check that there is a GPU device for Tensorflow to use
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
		""" Generate the array of all the images in the dataset and t array with the corresponding labels  """
		names = []  # not used, but just for seeing the file names

		for label in self.labels: # iterate through the dataset

			for filename in os.listdir(self.path+label+"/images/")[:3615]: # only 3615 files available in covid dataset, need to limit so that for loop doesnt try and iterate further since normal dataset has 10,000 imgs
				# divide by 255 to normalise the data
				file = str(f"{self.path}{label}/images/{filename}")
				arr = self.ImageToArray(file)
				# having issues with appending np array (it clears the array each time, so we conver to python list first and then make the whole thing a np array later
				arr = arr[::2, ::2].tolist() # Reduce size by factor of 2, convert to list so we can append and also shrink resolution to 150x150
				label_index = self.labels.index(label) # get the index of the label
				self.X.append(arr) # add to array
				self.y = np.append(self.y, label_index)
			print(f"DONE: {label}")

	def ProcessArrays(self):
		""" Process the training arrays to be compatible with the model """
		self.X = np.array(self.X) # convert to np array
		self.X = self.X/255 # normalise values to range from 0 to 1 so computation is easier. convert to float
		self.X = self.X.astype('float32') 	# convert to float

		self.y = np.array(self.y, dtype='int8') # convert to np array of int
		# y = tf.one_hot(y, 3)
		# one hot encode outputs
		self.y = np_utils.to_categorical(self.y) # convert labels vector to matrix of binary values

	def SplitDataset(self):
		"""Splits the dataset into training and testing sets.
		"""
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2) # split into 80% train, 20% test since we have small-ish dataset

	def BuildModel(self):
		"""Add layers to the model
		"""
		# print(self.labels)
		self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 150, 3))) # shape = X.shape[1:] # Add convolution layer
		self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(MaxPooling2D((2, 2))) # Add max pooling layer
		self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')) # Add convolution layers	
		self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(MaxPooling2D((2, 2))) # Add max pooling layer
		self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')) # Add convolution layers	
		self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		self.model.add(MaxPooling2D((2, 2))) # Add max pooling layer

		# example output part of the model
		self.model.add(Flatten()) # transform pooled feature map that is generated in the pooling step into a 1D vector
		self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform')) # dense layer
		self.model.add(Dense(len(self.labels), activation='softmax')) # final layer dense 2 since we have 2 labels

		# compile model
		self.model.compile( # compile the model
					  loss='binary_crossentropy',  # USE SPARSE if WE ARE USING THE ACTUAL LABEL NUMBERS E.G 1,2 BUT WE ALREADY CONVERTED TO CATEGORICAL
					  metrics=['accuracy'],
					  optimizer='adam'
					 )
		print(self.model.summary()) # print summary


	def TrainModel(self):
		""" Trains the model for 20 epochs using the dataset
		"""
		# print(self.X_train.shape)
		# print(self.y_train.shape)

		history = self.model.fit( # fit the model
			self.X_train,
			self.y_train,
			epochs=20,
			batch_size=32,
			validation_data=(self.X_test,self.y_test)
		)

	def EvaluateModel(self):
		"""Get loss value & metrics values for the model in test mode.
		"""
		evaluate = self.model.evaluate(self.X_test, self.y_test, verbose=0) #evaluate model
		print(evaluate)

	def SaveModel(self):
		"""Save the model so that it can be accessed later without having to retrain
		"""
#         NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"

		self.model.save(f"./6-conv-128-nodes-2-dense-{int(time.time())}.model") # save model to file

		f = open('labels.pickle', "wb")
		f.write(pickle.dumps(self.labels)) # serialises the labels so that it can be stored on disk and later deserialised for use.
		f.close()
    
if __name__ == '__main__':
	"""Driver function which is responsible for the overall creation of the model
	"""
	gen = ModelGenerator(['normal','covid'], "./dataset/")
	gen.CheckGPU()
	gen.ProcessImages()
	gen.ProcessArrays()
	gen.SplitDataset
	gen.BuildModel()
	gen.TrainModel()
	gen.EvaluateModel()
	gen.SaveModel()