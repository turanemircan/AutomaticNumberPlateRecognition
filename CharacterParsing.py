import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from LicensePlateDetection import plateDetection

data = os.listdir("Dataset Review") # klassördeki verileri aldık

name = data[1] # veriyi seçtik

img = cv2.imread("Dataset Review/"+name) # veriyi okuduk
img = cv2.resize(img,(500,500)) # veriyi yeniden boyutlandırdık

plate = plateDetection(img) # plaka tespiti yaptık
x,y,w,h = plate # plakanın koordinatlarını aldık
if(w>h):
    plate_bgr = img[y:y+h,x:x+w].copy()
else:
    plate_bgr = img[y:y+w,x:x+h].copy()

plt.imshow (plate_bgr) # plakayı gösterdik
plt.show()

# Görüntü boyutunu 2 katına çıkaracaz böylece daha iyi işlenebilecek

H,W = plate_bgr.shape[:2] # plakanın boyutlarını aldık
print("Orijinal boyut: ",W,H)

H,W = H*2,W*2
print("Yeni boyut: ",W,H)

plate_bgr = cv2.resize(plate_bgr,(W,H)) # plakayı boyutlandırdık

plt.imshow (plate_bgr) # plakayı gösterdik
plt.show()

# Plakayı gri tonlamalı hale getiriyoruz.
# plate_image = plaka işlem resmimiz

plate_image = cv2.cvtColor(plate_bgr,cv2.COLOR_BGR2GRAY) # plakayı gri tonlamalı hale getirdik

plt.title("Gri Tonlamalı Plaka")
plt.imshow(plate_image,cmap="gray") # plakayı gösterdik
plt.show()

th_image = cv2.adaptiveThreshold(plate_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2) 
# plakayı threshold uyguladık
# Eşikleme işlemi, bir resimdeki piksel değerlerini belirli bir eşik değerine göre ikiye ayırma işlemidir.

plt.title("Eşiklenmiş Plaka")
plt.imshow(th_image,cmap="gray")
plt.show()

kernel = np.ones((3,3),np.uint8) # kernel oluşturduk
th_image = cv2.morphologyEx(th_image,cv2.MORPH_OPEN,kernel,iterations=1)

plt.title("Gürültü Yok Edilmiş Plaka")
plt.imshow(th_image,cmap="gray")
plt.show()