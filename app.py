from flask import Flask, render_template, request
# from keras.models import load_model
from keras.preprocessing import image
from keras.utils import image_utils
import tensorflow as tf
from tensorflow import keras
from PIL import Image

#imported required packages
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


app = Flask(__name__)

dic = {0 : 'Normal', 1 : 'Pneumonia'}

model = tf.keras.models.load_model('trained.h5')

def predict_label(img_path):
	
	i =tf.keras.utils.load_img(img_path, target_size=(300,300))
	i = tf.keras.utils.img_to_array(i)/255.0
	i = i.reshape(1, 300,300,3)
	p = model.predict(i)
	if float(p[0]) > 0.5:
		p = 1
	else:
		p = 0
	return dic[p]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "This is Made by Trasun..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)
	
		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)