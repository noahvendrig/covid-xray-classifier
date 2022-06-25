from flask import Flask

from predict import predict
import cv2
from waitress import serve
import sys
import os

UPLOAD_FOLDER = 'static/files/'


templatesDir = './templates'
staticDir = './static'
app = Flask(__name__, template_folder=templatesDir, static_folder=staticDir)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

from pathlib import Path
import socket
# [f.unlink() for f in Path("./static/files").glob("*") if f.is_file()]  #remove all files in the folder

import os
# from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) # only log errors in flask app, nothing else so that console isn't cluttered

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def resize(im):
	# print(im)
	h, w, channels = im.shape
	max_w = 500
	ratio = h/w

	resized_h, resized_w = int(round(max_w*ratio)), max_w
	dims = (resized_w, resized_h)

	resized_im = cv2.resize(im, dims)

	return resized_im

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('index.html', )

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':	
		flash('No image selected for uploading')
		return redirect(request.url)

	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

		im = cv2.imread(path) # read 
		os.remove(path)
		im = resize(im)
		cv2.imwrite(path, im)

		im = [im]
		print(f"Image Saved To {path}")
		prediction = predict([path])
		print(f"Prediction: {prediction}")
		return render_template('index.html', filename=filename, prediction=prediction)

	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='files/' + filename), code=301)

    
def application():
	port = 5000
	hostname=socket.gethostname()
	ip_addr=socket.gethostbyname(hostname)

	print(f"App Hosted at {ip_addr}:{port}") #indicate where the app is hosted
	# serve(app,  host="0.0.0.0", port=port)

	app.run(host='0.0.0.0', debug=False)
    # app.run(debug=True, host="0.0.0.0") # only for development
	

# application() # not needed since running from cli.py