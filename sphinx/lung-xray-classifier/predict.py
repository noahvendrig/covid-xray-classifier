import numpy as np
import cv2
import tensorflow as tf
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Classifier:
    """_summary_
    """
    def __init__(self, model_path, img_paths):
        """_summary_

        Args:
            model_path (_type_): _description_
            img_paths (_type_): _description_
        """
        self.model = tf.keras.models.load_model(model_path)
        self.labels = pickle.loads(open('labels.pickle', "rb").read())
        for img in img_paths:
            self.Classify(img)


    def ImageToArray(self, file):
        """_summary_

        Args:
            file (_type_): _description_
        """
           # reads an image in the BGR format
        try:
            img_arr = cv2.imread(file)
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)   # BGR -> RGB
            return img_arr
        except Exception as e:
            print("INCORRECT FILE PATH")
            os._exit(0) # exit the program but no error message to clog terminal when debuggig

    
    def Classify(self, img):
        """_summary_

        Args:
            img (_type_): _description_

        Returns:
            _type_: _description_
        """
        # actual_label = img.split("/")[-1]
        img_arr = self.ImageToArray(img)
        new_img = cv2.resize(img_arr, (150,150))
        # new_img = (new_img[::2, ::2]/255).astype('float32')
        new_img = (new_img/255).astype('float32')
        new_img = tf.expand_dims(new_img, 0) # model expects a dataset so we expand_dims
        print()

        try:
            self.prediction = self.model.predict(new_img) # returns list with probabilities of being each label
            # print(self.prediction[0])
            self.prediction = np.argmax(self.prediction, axis=None, out=None) # convert from categorical back to index
            # print(f"Predicted :- {self.labels[self.prediction]}")
            # print(f"Actual :- {actual_label}")
            # print()
            self.prediction = self.labels[self.prediction]
            print()
        except:
            return "Error in Model Prediction", os._exit(0)


def predict(input):
    """_summary_

    Args:
        input (_type_): _description_

    Returns:
        _type_: _description_
    """
#     print("RUNNING MAIN")
    # if __name__ == '__main__':
    # Only executed if you start this script as the main script,
    # i.e. you enter 'python path/to/main.py' in a terminal.
    # Assuming you saved the script in the directory 'path/to'
    # and named it 'main.py'.
    # c = Classifier("6-conv-128-nodes-2-dense-1654694547.model", ["./dataset/normal/images/Normal-10000.png" , "./dataset/covid/images/COVID-3615.png"])


    # res = Classifier("6-conv-128-nodes-2-dense-1655171754.model", input)
    res = Classifier("6-conv-128-nodes-2-dense-1656148579.model", input)
    return res.prediction
# main(["./dataset/normal/images/Normal-10000.png"])

# print(main())
# print(predict(["./dataset/normal/images/Normal-10000.png"]))