from flask import Flask, redirect, url_for, render_template, request, redirect
import os
import cv2
from PIL import Image
from utils.preprocessing import PreProcess
os.environ['TF_KERAS'] = '1'
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import random

from imutils import paths
import numpy as np
import pandas as pd
import skimage as sk
import skimage.transform
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
import ssl

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

print('Started')
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ssl._create_default_https_context = ssl._create_unverified_context

def generate_vgg_model(classes_len: int):
    """
    Function to create a VGG19 model pre-trained with custom FC Layers.
    If the "advanced" command line argument is selected, adds an extra convolutional layer with extra filters to support
    larger images.
    :param classes_len: The number of classes (labels).
    :return: The VGG19 model.
    """
    img_input = Input(shape=(512, 512, 1))
    img_conc = Concatenate()([img_input, img_input, img_input])

    model_base = VGG19(include_top=False, weights='imagenet', input_tensor=img_conc)

    model = Sequential()
    model.add(model_base)

    if model == "advanced":
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding='same'))
        model.add(Conv2D(512, (3, 3),
                         activation='relu',
                         padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=512, activation='relu', name='Dense_Intermediate_1'))
    model.add(Dense(units=32, activation='relu', name='Dense_Intermediate_2'))

   
    if classes_len == 2:
        model.add(Dense(1, activation='sigmoid', name='Output'))
    else:
        model.add(Dense(classes_len, activation='softmax', name='Output'))


    return model

model = generate_vgg_model(3)

model.load_weights(r'D:\Swaminathan_chellappa_stuff\FYP2022\flask_trial\model_files\vgg_19_p')






app = Flask(__name__)


app.config['IMAGE_UPLOADS'] = r'D:\Swaminathan_chellappa_stuff\FYP2022\flask_trial\uploads'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = ["PNG"]
app.config['PREPROCESS_IMAGES'] = r'D:\Swaminathan_chellappa_stuff\FYP2022\flask_trial\processed'
app.config['DISPLAY_IMAGE'] = r'D:\Swaminathan_chellappa_stuff\FYP2022\flask_trial\static'

def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit('.')[1]

    if ext.upper() in app.config['ALLOWED_IMAGE_EXTENSIONS']:
        return True
    else:
        return False 



@app.route('/')
def home():
    return render_template("index.html")

@app.route('/aboutus')
def about_us():
    return render_template("about.html")

@app.route('/upload-image',methods=["GET","POST"])
def upload_image():

    if request.method =='POST':

        if request.files:

            image = request.files['image']

            if image.filename == '':
                print("Image must have a filename")
                return redirect(request.url)
            
            if not allowed_image(image.filename):
                print('That image extension is not allowed')
                return redirect(request.url)

            img_path = os.path.join(app.config['IMAGE_UPLOADS'],image.filename)
            image.save(img_path)
            
            f_name = 'display.jpeg'
            img_path_display = os.path.join(app.config['DISPLAY_IMAGE'],f_name)
            preprocess_img_path_display = os.path.join(app.config['DISPLAY_IMAGE'],'preprocessed.jpeg')
            img = Image.open(img_path)
            rgb_img = img.convert('RGB')
            rgb_img.save(img_path_display)

            #image.save(os.path.join(app.config['IMAGE_UPLOADS'],image.filename))
            img_save_path = os.path.join(app.config['PREPROCESS_IMAGES'],image.filename)
            print('Image saved')

            prep_obj = PreProcess(img_path)
            preprocessed_img = prep_obj.run(img_path)
            print('preprocessed_img shape',preprocessed_img.shape)

            cv2.imwrite(img_save_path,preprocessed_img)
            cv2.imwrite(preprocess_img_path_display,preprocessed_img)
            test_img = cv2.imread(img_save_path)
            print('Reloaded_shape : ',test_img.shape)
            image = load_img(img_save_path,
                     color_mode="grayscale",
                     target_size=(512, 512))
            print('Time to Predict')
            image = img_to_array(image)
            image /= 255.0
            image=np.reshape(image,(-1,512,512,1))
            y=model.predict(image)
            y=np.argmax(y)

            prediction = ''
            if(np.equal(y,[0])):
                prediction = 'Benign'
                print("benign")
            elif(np.equal(y,[1])):
                prediction = 'Malignant'
                print("malignant")
            else:
                prediction = 'Normal'
                print("normal")

            return render_template("display_result.html",prediction=prediction,img_path=img_path_display)
            #return redirect(request.url)
    return render_template("upload_image.html")





if __name__ == "__main__":
    app.run(debug=True)