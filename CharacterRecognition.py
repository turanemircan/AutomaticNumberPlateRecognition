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

urls = []
classLabels = []
for cls in classes:
    images = os.listdir(path + cls)
    for image in images:
        urls.append(path + cls + "/" + image)
        classLabels.append(cls)
        singleBatch += 1

df = pd.DataFrame({"address": urls, "class": classLabels})

def process(img):
    
    newShape = img.reshape((1600, 5, 5))
    means = []
    for part in newShape:
        mean = np.mean(part)
        means.append(mean)
    means = np.array(means)
    means = means.reshape(1600,)
    return means

def preprocess(img):
    return img / 255

target_size = (200, 200) 
batch_size = singleBatch 

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess)

train_set = train_gen.flow_from_dataframe(df, x_col="address", y_col="class",
                                          target_size=target_size,
                                          color_mode="grayscale",
                                          shuffle=True,
                                          class_mode='sparse',
                                          batch_size=batch_size)

images, train_y = next(train_set)
train_x = np.array(list(map(process, images))).astype("float32")
train_y = train_y.astype(int)

print("Random Forest / Rassal Orman Modeli Eğitiliyor...")

randomForestClass = RandomForestClassifier(n_estimators=10,criterion="entropy")

randomForestClass.fit(train_x,train_y)

print("Random Forest / Rassal Orman Modeli Eğitildi.")

pred = randomForestClass.predict(train_x)

acc = accuracy_score(pred,train_y)

print("Random Forest / Rassal Orman Modeli Doğruluk Oranı: ",acc)

file = "RandomForestModel.rfc"

pickle.dump(randomForestClass,open(file, "wb")) # Modeli şifreli kaydetme işlemi

# dosyayı okumak için pickle.load(open(file, "rb")) kullanılır