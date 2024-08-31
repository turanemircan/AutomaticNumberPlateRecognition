import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from LicensePlateDetection import plateDetection
from LicensePlateRecognition import plateRecognition

dataAdress = os.listdir("Dataset Review")

name = dataAdress[3]

print("Image:","Dataset Review/"+name)
img = cv2.imread("Dataset Review/"+name)
img = cv2.resize(img,(500,500))

plate = plateDetection(img)
plateImg,plateChar = plateRecognition(img,plate)
print("Plate in the image:",plateChar)