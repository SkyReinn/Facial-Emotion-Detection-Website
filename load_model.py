# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:24:08 2023
@author: David Wang
"""

import zipfile
import tensorflow as tf

#unzip the cnn model from google drive
cnn_path = "C:\\Users\\David Wang\\Downloads\\cnn.zip"
with zipfile.ZipFile(cnn_path, 'r') as zip_ref:
    zip_ref.extractall('')

#load the cnn model and save it as an h5 file for easy access
cnn = tf.keras.models.load_model('cnn')
cnn.save('cnn_model.h5', save_format='h5')

#unzip the vgg model from google drive
vgg_path = "C:\\Users\\David Wang\\Downloads\\vgg.zip"
with zipfile.ZipFile(vgg_path, 'r') as zip_ref:
    zip_ref.extractall('')

#load the cnn model and save it as an h5 file for easy access
vgg = tf.keras.models.load_model('vgg')
vgg.save('vgg_model.h5', save_format='h5')
