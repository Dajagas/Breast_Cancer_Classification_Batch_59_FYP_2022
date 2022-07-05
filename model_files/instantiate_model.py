#!/usr/bin/env python
# coding: utf-8
import os 
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

image = load_img(r'D:\Swaminathan_chellappa_stuff\7th sem\Final Year Project\flask_trial\model_files\test_1.jpeg',
                     color_mode="grayscale",
                     target_size=(512, 512))
image = img_to_array(image)
image /= 255.0
    
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