from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from gtts import gTTS 
from playsound import playsound   
# This module is imported so that we can  
# play the converted audio 
import os 
def extract_face(filename, required_size=(96, 96)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	return image
import glob
print('hello')

#for file in glob.glob("C:/Users/HP/Downloads/images/Train/*"):
    #print(c)
    #file = "C:/Users/HP/Downloads/images/Train/Murtuza.jpeg"
    #person_name = os.path.splitext(os.path.basename(file))[0]
  #  image_file = cv2.imread(file, 1)
   # im1 = extract_face(file,(96,96))
    #im1 = im1.save("train/"+person_name+".jpg") 
c=1
for file in glob.glob("C:/Users/HP/Downloads/images/Test/*"):
    #print(c)
    #file = "C:/Users/HP/Downloads/images/Train/Murtuza.jpeg"
    person_name = os.path.splitext(os.path.basename(file))[0]
    print(person_name)
    image_file = cv2.imread(file, 1)
    im1 = extract_face(file,(96,96))
    im1 = im1.save("test/testcase"+str(c)+".jpg") 
    c = c +1
    