from distutils.core import setup # Need this to handle modules
import py2exe 
from flask import Flask
from predict import predict
import cv2
from app import application
import numpy as np
import cv2
import tensorflow as tf
import pickle
import os

setup(console=['cli.py']) # Calls setup function to indicate that we're dealing with a single console application