import cv2
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

path = "Character Set/"
classes = os.listdir(path)
singleBatch = 0

urls= [] # Karakter resimlerinin adreslerini tutacak
grades = []

for grade in classes:
    images = os.listdir(path+grade)
    for image in images:
        urls.append(path+grade+"/"+image)
        grades.append(grade)
        singleBatch += 1

dataFrame = pd.DataFrame({"URL":urls,"Grade":grades})

def process(img):
    newHeight = img.reshape((1600,5,5))
    averages = []
    for piece in newHeight:
        average = np.mean(piece)
        averages.append(average)
    averages = np.array(averages)
    averages = averages.reshape(1600,)
    return averages
    
def pretreatment(img):
    return img/255

targetSize = (200,200)
batchSize = singleBatch
