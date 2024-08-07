import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from LicensePlateDetection import plateDetection

data = os.listdir("Dataset Review") # klassördeki verileri aldık

name = data[3] # veriyi seçtik

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
print("Original size: ",W,H)

H,W = H*2,W*2
print("New size: ",W,H)

plate_bgr = cv2.resize(plate_bgr,(W,H)) # plakayı boyutlandırdık

plt.imshow (plate_bgr) # plakayı gösterdik
plt.show()

# Plakayı gri tonlamalı hale getiriyoruz.
# plate_image = plaka işlem resmimiz

plate_image = cv2.cvtColor(plate_bgr,cv2.COLOR_BGR2GRAY) # plakayı gri tonlamalı hale getirdik

plt.title("Grayscale Plate")
plt.imshow(plate_image,cmap="gray") # plakayı gösterdik
plt.show()

th_image = cv2.adaptiveThreshold(plate_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2) 
# plakayı threshold uyguladık
# Adaptive thresholding, bir resimdeki piksel değerlerini belirli bir eşik değerine göre ikiye ayırma işlemidir.
# Eşikleme işlemi, bir resimdeki piksel değerlerini belirli bir eşik değerine göre ikiye ayırma işlemidir.

plt.title("Thresholded Plate")
plt.imshow(th_image,cmap="gray")
plt.show()

kernel = np.ones((3,3),np.uint8) # kernel oluşturduk
th_image = cv2.morphologyEx(th_image,cv2.MORPH_OPEN,kernel,iterations=1)

plt.title("Noise Cancelled Plate")
plt.imshow(th_image,cmap="gray")
plt.show()

cnt = cv2.findContours(th_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = cnt[0]
cnt = sorted(cnt,key=cv2.contourArea,reverse=True)[:15] # konturları sıraladık

for i,c in enumerate(cnt):
    rectanial = cv2.minAreaRect(c) # konturların dış dikdörtgenini aldık
    (x,y),(w,h),r = rectanial # koordinatları aldık
    
    control1 = max([w,h]) <W/4
    control2 = w * h > 200
    
    if(control1 and control2):
        print("character ->",x,y,w,h)
        
        box = cv2.boxPoints(rectanial)
        box = np.int64(box)
        
        minx = np.min(box[:,0])
        miny = np.min(box[:,1])
        maxx = np.max(box[:,0])
        maxy = np.max(box[:,1])
        
        focus = 2
        
        minx = max(0,minx-focus)
        miny = max(0,miny-focus)
        maxx = min(W,maxx+focus)
        maxy = min(H,maxy+focus)
        
        cut = plate_bgr[miny:maxy,minx:maxx].copy()
        
        try:
            cv2.imwrite(f"Character Set/{name}_{i}.jpg",cut)
        except:
            pass
        
        write = plate_bgr.copy()
        cv2.drawContours(write,[box],0,(0,255,0),1)
        
        plt.imshow(write)
        plt.show()    