import numpy as np
import cv2
import tensorflow as tf
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Classifier:

    def __init__(self, model_path, img_paths):
        self.model = tf.keras.models.load_model(model_path)
        self.labels = pickle.loads(open('labels.pickle', "rb").read())
        for img in img_paths:
            self.Classify(img)


    def ImageToArray(self, file):
        img_arr = cv2.imread(file)   # reads an image in the BGR format
        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)   # BGR -> RGB
            return img_arr
        except Exception as e:
            print("INCORRECT FILE PATH")
            return
        
    
    
    def Classify(self, img):
        actual_label = img.split("/")[-1]
        new_img = self.ImageToArray(img)
        new_img = (new_img[::2, ::2]/255).astype('float32')
        new_img = tf.expand_dims(new_img, 0) # model expects a dataset so we expand_dims
        self.prediction = self.model.predict(new_img)
        self.prediction = np.argmax(self.prediction, axis=None, out=None) # convert from categorical back to index
        
        # print(f"Predicted :- {self.labels[self.prediction]}")
        # print(f"Actual :- {actual_label}")
        # print()
        self.prediction = self.labels[self.prediction]

def main():
    print("RUNNING MAIN")
    # if __name__ == '__main__':
    # Only executed if you start this script as the main script,
    # i.e. you enter 'python path/to/main.py' in a terminal.
    # Assuming you saved the script in the directory 'path/to'
    # and named it 'main.py'.
    # c = Classifier("6-conv-128-nodes-2-dense-1654694547.model", ["./dataset/normal/images/Normal-10000.png" , "./dataset/covid/images/COVID-3615.png"])
    res = Classifier("6-conv-128-nodes-2-dense-1655171754.model", ["./dataset/normal/images/Normal-10000.png"])
    return res.prediction
# main()

print(main())