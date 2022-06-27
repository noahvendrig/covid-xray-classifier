from predict import predict
import cv2
from waitress import serve
import sys
import os
from threading import Timer
import webbrowser

UPLOAD_FOLDER = 'static/files/' # folder to store uploaded images

from pathlib import Path
import socket
# [f.unlink() for f in Path("./static/files").glob("*") if f.is_file()]  #remove all files in the folder

import os
# from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template # import flask components
from werkzeug.utils import secure_filename

import logging

templatesDir = './templates' # directory where the templates are stored
staticDir = './static' # directory for static files
app = Flask(__name__, template_folder=templatesDir, static_folder=staticDir) # create the Flask app

app.secret_key = "secret key" # needed for flask sessions
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # set the upload folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) # only log errors in flask app, nothing else so that console isn't cluttered

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg']) # will only accept the following image types

def resize(im):
	"""Resizes image to a width of 500 so that it can be displayed on the webpage

	Args:
		im (numpy arr): Input image as an array

	Returns:
		numpy arr: Resized image
	"""
	h, w, channels = im.shape # get the height, width and channels of the image
	max_w = 500 # maximum width
	ratio = h/w # ratio of height to width

	resized_h, resized_w = int(round(max_w*ratio)), max_w
	dims = (resized_w, resized_h) # get dimensions that retain image's aspect ratio but have maximum width of 500

	resized_im = cv2.resize(im, dims) # resize image to specified dimensions
	return resized_im

def allowed_file(filename):
	"""Check if file is allowed to be uploaded (filetypes is either png, jpg or jpeg)

	Args:
		filename (str): name of the file to be uploaded (e.g. myimage.jpg)

	Returns:
		bool: Whether the file is allowed to be uploaded or not
	"""
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	"""Displays the main page with the file upload form
	"""
	return render_template('index.html', )

@app.route('/', methods=['POST'])
def upload_image():
	"""Upload the image selected by the user

	Returns:
		str: filename of the image uploaded
		str: predictions of the image uploaded
	"""
	if 'file' not in request.files:
		flash('No file part') # display error message
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':	
		flash('No image selected for uploading') # display error message
		return redirect(request.url)

	if file and allowed_file(file.filename): # if the file is allowed and has been uploaded 
		filename = secure_filename(file.filename)
		path = os.path.join(app.config['UPLOAD_FOLDER'], filename) # save the file to the upload folder
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
		im = cv2.imread(path) # read 
		os.remove(path) #delete the old file
		im = resize(im) # resize the image to be displayed
		cv2.imwrite(path, im) # save the image locally
		print(f"Image Saved To {path}")
		prediction = predict([path]) # predict the image
		print(f"Prediction: {prediction}")
		return render_template('index.html', filename=filename, prediction=prediction)

	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif') # display error message if the filetype is not allowed
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):

	"""Display the uploaded image

	Args:
		filename (str): Filename of uploaded file
	"""
	return redirect(url_for('static', filename='files/' + filename), code=301)

@app.route('/docs')
@app.route('/')
def docs():
	"""Redirect the user to the documentation page (/docs)
	"""
	return redirect(url_for('static', filename='build/html/index.html'), code=302)

def application():
	"""Host the App on local IP Address (port 5000 by default)
	"""
	port = 5000
	hostname=socket.gethostname()	
	ip_addr=socket.gethostbyname(hostname)

	webbrowser.open(f"http://{ip_addr}:{port}", new=1) # opens the app in a new browser window
	print(f"App Running at: {ip_addr}:{port}") # indicate where the app is hosted
	try:
		serve(app,  host="0.0.0.0", port=5000)
		# app.run(debug=True, host="0.0.0.0") # only for development
	except Exception as e:
		print(e)
	

# application() # not needed since running from cli.py