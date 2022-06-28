from predict import predict # v1.0.0 by Noah Vendrig (06/2022) - predict function from './predict.py'
import cv2 # v4.5.5 by Intel Corporation, Willow Garage, Itseez (06/2000) - Used for image processing: converting RGB images to numpy arrays and resizing images so that they are compatible with the model.
from waitress import serve # v2.1.2 by Zope Foundation and Contributors (30/12/2011) - Production WSGI server for the web app.
import os
import webbrowser

UPLOAD_FOLDER = 'static/files/' # folder to store uploaded images

import socket

#from pathlib import Path
# [f.unlink() for f in Path("./static/files").glob("*") if f.is_file()]  #remove all files in the folder not needed right now

import os
from flask import Flask, flash, request, redirect, url_for, render_template # v2.1.2 by Armin Ronacher (01/04/2010) - Allows for the creation of the GUI as a web application hosted on local IP
from werkzeug.utils import secure_filename # v2.1.2 by Armin Ronacher (10/12/2007) - Create WSGI server for Flask

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
		im = cv2.imread(path) # read image from disk into array
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
	port = 5000 # my favourite port
	hostname=socket.gethostname()	
	ip_addr=socket.gethostbyname(hostname) # get ip of the user's machine

	webbrowser.open(f"http://{ip_addr}:{port}", new=1) # opens the app in a new browser window
	print(f"App Running at: {ip_addr}:{port}") # indicate where the app is hosted
	try:
		serve(app,  host="0.0.0.0", port=5000) # for production
		# app.run(debug=True, host="0.0.0.0") # only for development
	except Exception as e:
		print(e) # oh no there was an error!!
	

# application() # not needed since running from cli.py