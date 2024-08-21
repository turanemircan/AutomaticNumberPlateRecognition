import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

def process(img):
    
    newShape = img.reshape((1600, 5, 5))
    means = []
    for part in newShape:
        mean = np.mean(part)
        means.append(mean)
    means = np.array(means)
    means = means.reshape(1600,)
    return means

path = "Character Set/"
classes = os.listdir(path)
singleBatch = 0

urls = []
classLabels = []

for cls in classes:
    images = os.listdir(path + cls)
    for image in images:
        urls.append(path + cls + "/" + image)
        classLabels.append(cls)
        singleBatch += 1

df = pd.DataFrame({"address": urls, "classs": classLabels})

classLabels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10,
          'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
          'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30,
          'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'arkaplan': 36}

file = "RandomForestModel.rfc"

randomForestClass = pickle.load(open(file, 'rb')) # read byte

index = list(classLabels.values()) # [0, 1, 2,..., 35, 36]

classes = list(classLabels.keys()) # ['0', '1', '2',..., 'Z', 'arkaplan']

df = df.sample(frac=1)

for address,classs in df.values:
    img = cv2.imread(address, 0)
    image = cv2.resize(img, (200, 200))
    image = image / 255
    attributes = process(image) 
    
    result = randomForestClass.predict([attributes]) [0]
    print("result:", result)
    
    ind = index.index(result)
    classs = classes[ind]
    plt.imshow(image,cmap="gray")
    plt.title(f"Predicted: {classs}")
    plt.show()