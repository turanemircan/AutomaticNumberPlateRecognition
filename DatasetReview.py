import os
import matplotlib.pyplot as plt
import cv2
from LicensePlateDetection import plateDetection

'''

data = os.listdir("Dataset Review")

for image_url in data:
    img = cv2.imread("Dataset Review/"+image_url)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(500,500))
    plt.imshow(img)
    plt.show()

'''    

data = os.listdir("Dataset Review")

for image_url in data:
    img = cv2.imread("Dataset Review/"+image_url)
    
    img = cv2.resize(img,(500,500))
    plate = plateDetection(img) # Plakanın koordinatlarını öğrendik
    
    x,y,w,h = plate
    
    if(w>h):
        plate_bgr = img[y:y+h,x:x+w].copy()
    else:
        plate_bgr = img[y:y+w,x:x+h].copy()
            
    img = cv2.cvtColor(plate_bgr,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()