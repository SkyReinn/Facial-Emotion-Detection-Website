# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:49:37 2023
@author: David Wang
"""


from flask import Flask, request, render_template
from keras.models import load_model
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np

#load vgg model
vgg = load_model('vgg_model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #load image
    image = request.files['image']
    img = Image.open(image)
    
    #resize image
    img = img.resize((48, 48))
    
    #convert to grayscale
    img = img.convert('L')
    
    #save grayscale image to variable
    img_array = np.array(img)
    
    #show image in 48*48 greyscale format
    plt.imshow(img_array, cmap='gray')
    plt.show()
    
    #reshape image for prediction
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.stack((img_array,)*3, axis=-1)
    img_array = np.reshape(img_array, (1, 48, 48, 3))

    #preciction
    pred = vgg.predict(img_array)[0]
    emotions = ["Angry","Happy","Sad","Surprise","Neutral"]
    pred_emotion = emotions[np.argmax(pred)]
    print("prediction:")
    print(pred_emotion)
    print(pred)
    
    #return prediction
    return render_template('index.html', prediction_text = pred_emotion)


if __name__ == "__main__":
    app.run()
