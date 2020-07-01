from gtts import gTTS 
from playsound import playsound  
import cv2
# This module is imported so that we can  
# play the converted audio 
import os 
def sound(message,identity):
# The text that you want to convert to audio 
    mytext = message
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False) 
    path = identity + ".mp3"
    myobj.save(path) 
    playsound(path)
# Playing the converted file 
try:
    img1 = cv2.imread("images/danielle.png", 1)
    image = cv2.resize(img1, (96, 96))

except Exception as e:
    print(str(e))