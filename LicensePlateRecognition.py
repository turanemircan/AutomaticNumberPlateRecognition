import cv2
import numpy as np
import pickle

file = "RandomForestModel.rfc"
randomForestClass = pickle.load(open(file, 'rb'))

classLabels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10,
          'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
          'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30,
          'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'arkaplan': 36}

index = list(classLabels.values()) # [0, 1, 2,..., 35, 36]
classes = list(classLabels.keys())

def process(img):
    
    newShape = img.reshape((1600, 5, 5))
    means = []
    for part in newShape:
        mean = np.mean(part)
        means.append(mean)
    means = np.array(means)
    means = means.reshape(1600,)
    return means

def plateRecognition(img,plate):
    x,y,w,h = plate # plakanın koordinatlarını aldık
    if(w>h):
        plate_bgr = img[y:y+h,x:x+w].copy()
    else:
        plate_bgr = img[y:y+w,x:x+h].copy()

    H,W = plate_bgr.shape[:2] # plakanın boyutlarını aldık
    H,W = H*2,W*2

    plate_bgr = cv2.resize(plate_bgr,(W,H)) # plakayı boyutlandırdık

    # Plakayı gri tonlamalı hale getiriyoruz.
    # plate_image = plaka işlem resmimiz

    plate_image = cv2.cvtColor(plate_bgr,cv2.COLOR_BGR2GRAY) # plakayı gri tonlamalı hale getirdik

    th_image = cv2.adaptiveThreshold(plate_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2) 

    kernel = np.ones((3,3),np.uint8) # kernel oluşturduk
    th_image = cv2.morphologyEx(th_image,cv2.MORPH_OPEN,kernel,iterations=1)

    cnt = cv2.findContours(th_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0]
    cnt = sorted(cnt,key=cv2.contourArea,reverse=True)[:15]

    write = plate_bgr.copy()
    
    currentPlate = [] # cls, minx
    
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
            
            diagnosis = cv2.cvtColor(cut,cv2.COLOR_BGR2GRAY)
            diagnosis = cv2.resize(diagnosis,(200,200))
            diagnosis = diagnosis/255
            attributes = process(diagnosis)
            character = randomForestClass.predict([attributes])[0]
            ind = index.index(character)
            cls = classes[ind]
            
            if cls=="background":
                continue
            
            currentPlate.append([cls,minx])
            cv2.putText(write,cls,(minx-2,miny-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)           
            cv2.drawContours(write,[box],0,(0,255,0),1) 
    return write,currentPlate