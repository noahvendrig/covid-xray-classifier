# Required modules 
import numpy as np
import cv2
import tensorflow as tf
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # tensorflow INFO, WARNING, and ERROR messages are not printed

class Classifier:
    """Classifier Class used to evaluate lung-xray images into either covid or normal
    """
    def __init__(self, model_path, img_paths):
        """Initialise the Classifier class' attributes

        Args:
            model_path (str): Path to the CNN Model created in Tensorflow
            img_paths (list): list of paths to the image to be evaluated by the model. Currently the list is only of one element
        """
        self.model = tf.keras.models.load_model(model_path) # Loads a compiled Keras model instance
        self.labels = pickle.loads(open('labels.pickle', "rb").read()) # deserialise the pickle object so that the labels can be used in script
        for img in img_paths: # iterate through list of img paths
            self.Classify(img) # call the classify function for every image inputted (currently only one image)


    def ImageToArray(self, file):
        """Converts input image into a numpy array

        Args:
            file (str): path to the image to be converted to numpy array

        Returns:
            img_arr (numpy arr): numpy array of the image
        """
           # reads an image in the BGR format
        try:
            img_arr = cv2.imread(file) # returns a 3d array
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)   # convert BGR -> RGB
            return img_arr
        except Exception as e:
            print("INCORRECT FILE PATH") # incase there are issues with the file path 
            os._exit(0) # exit the program but no error message to clog terminal when debuggig

    
    def Classify(self, img):
        #
        """Generates a prediction for the submitted image

        Args:
            img (str): Path to the image to be classified
        """
        # actual_label = img.split("/")[-1]
        img_arr = self.ImageToArray(img) # convert to numpy array
        new_img = cv2.resize(img_arr, (150,150)) # resize image so that it can be used for training
        # new_img = (new_img[::2, ::2]/255).astype('float32')
        new_img = (new_img/255).astype('float32') # normalise values to range from 0 to 1 so computation is easier. convert to float
        new_img = tf.expand_dims(new_img, 0) # model expects a dataset so we expand_dims. shape goes from (150,150,3) --> TensorShape([1,150,150,3])

        try:
            self.prediction = self.model.predict(new_img) # returns list with probabilities of being each label
            # print(self.prediction[0])
            self.prediction = np.argmax(self.prediction, axis=None, out=None) # convert from categorical back to index

            self.prediction = self.labels[self.prediction] # get the string from the index so it can be displayed

        except:
            return "Error in Model Prediction", os._exit(0) # display error message


def predict(input):
    """This is the main function that drives the generation of the prediction

    Args:
        input (list): list of paths to the image to be evaluated by the model. Currently the list is only of one element but can be further expanded to evaluate multiple images

    Returns:
        str: 'prediction' attribute of the Classifier Instance, which is the model prediction of either 'Normal' or 'Covid'
    """
    # c = Classifier("6-conv-128-nodes-2-dense-1654694547.model", ["./dataset/normal/images/Normal-10000.png" , "./dataset/covid/images/COVID-3615.png"])

    # res = Classifier("6-conv-128-nodes-2-dense-1655171754.model", input)
    res = Classifier("6-conv-128-nodes-2-dense-1656148579.model", input) # create instance of the Classifier class that uses the previously trained model

    return res.prediction # return the prediction made by the model to the UI

# print(predict(["./dataset/normal/images/Normal-10000.png"]))