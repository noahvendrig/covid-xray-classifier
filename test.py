from flask import Flask

from predict import main
import cv2


UPLOAD_FOLDER = 'static/files/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

from pathlib import Path

# [f.unlink() for f in Path("./static/files").glob("*") if f.is_file()]  #remove all files in the folder

import os
# from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def resize(im):
	print(im)
	h, w, channels = im.shape

	max_h = int(400)
	ratio = w/h
	resized_w, resized_h = int(round(max_h*ratio)), max_h
	dims = (resized_w, resized_h)
	# resized_im = im.resize(im, (resized_w, resized_h))
	resized_im = cv2.resize(im, dims)
	print(resized_im.shape)
	return resized_im

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('i.html', )

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
		file.save(path)
		#print('upload_image filename: ' + filename)
		im = cv2.imread(path) # read 
		os.remove(path)
		im = resize(im)
		cv2.imwrite(path, im)
		# flash('Image successfully uploaded and displayed below')
		#["./dataset/normal/images/Normal-10000.png"]
		im = [im]
		print(path)
		prediction = main([path])
		return render_template('i.html', filename=filename, prediction=prediction)

	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='files/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")