import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

'''

dataAdress = os.listdir("Dataset Review")

img = cv2.imread("Dataset Review/"+dataAdress[0])
img = cv2.resize(img,(500,500))

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # OpenCV reads images as BGR, but Matplotlib reads images as RGB
plt.show()

img_bgr = img
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Convert image to grayscale

plt.imshow(img_gray,cmap="gray")
plt.show()

# Kenar tespiti
# transaction image ti_img

# Kenarlıkları daha belirgin hale getirmek için median blur uygulandı
# Median blur resimi 5x5 bir kernel(boyut) ile filtreler ve her pikselin değerini ortanca değer ile değiştirir

ti_img = cv2.medianBlur(img_gray,5) # Apply median blur to the image
ti_img = cv2.medianBlur(ti_img,5) # Apply median blur to the image

plt.imshow(img_gray,cmap="gray")
plt.show()

# Yoğunluk merkezi almak için median(ortanca değeri) değeri hesaplandı
# Kenar tespiti için alt ve üst yoğunluk merkezi belirlendi
# Canny algoritması ile kenar tespiti yapıldı

median = np.median(ti_img)

low = 0.67 * median # 3/2 alt yoğunluk merkezi
high = 1.33 * median # 4/3 üst yoğunluk merkezi

border = cv2.Canny(ti_img,low,high) # Apply Canny edge detection to the image

plt.imshow(border,cmap="gray")
plt.show()

# Kenar tespiti daha iyi yapılabilmesi için genişletme işlemi yapılması lazım

border = cv2.dilate(border,np.ones((3,3),np.uint8),iterations=1) # Apply dilation to the image

plt.imshow(border,cmap="gray")
plt.show()

# Aynı piksel değerlerine sahip piksellerin bir araya toplanmasına contour denir.
# Contour tespiti yapıldı

# RETR_TREE: Tüm hiyerarşik yapıyı döndürür
# CHAIN_APPROX_SIMPLE: Kenar noktalarını sıkıştırır ve sadece kenar noktalarını döndürür

cnt = cv2.findContours(border,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # Find contours in the image
cnt = cnt[0]
cnt = sorted(cnt,key=cv2.contourArea,reverse=True) # [:10] # Sort the contours by area and get the largest 10 contours

H,W = 500,500
plate = None

for c in cnt:
    rectanial = cv2.minAreaRect(c) # Dikdörtgen yapısını alır
    (x,y),(w,h),r = rectanial # r: döndürme açısı
    if(w>h and w>h*2) or (h>w and h>w*2) : # Plaka boyutu belirlendi
        box = cv2.boxPoints(rectanial) # Dikdörtgenin köşe noktalarını alır # [[0,1],[12,13],[14.30],[25,23]]
        box = np.int64(box) # 64 almamızın sebebi köşe noktalarının bazen resmin dışında kalıp - değer alabiliyor o yüzden
        # tam sayı alırız
        
        minx = np.min(box[:,0])
        miny = np.min(box[:,1])
        maxx = np.max(box[:,0])
        maxy = np.max(box[:,1])
        
        potantial_plate = img_gray[miny:maxy,minx:maxx].copy() # copy dememizin sebebi orjinal resmi bozmamak
        potantial_median = np.median(potantial_plate)
        
        control1 = potantial_median > 84 and potantial_median < 200 # Yoğunluk merkezi kontrolü
        control2 = h < 50 and w < 150 # Sınır kontrolü
        control3 = w < 50 and h < 150 # Sınır kontrolü
        
        print(f"Median: {potantial_median} - Width: {w} - Height: {h}")
        
        plt.figure()
        control = False
        if (control1 and (control2 or control3)):
            # Plaka tespit edildi
            
            cv2.drawContours(img,[box],0,(0,255,0),2) # ilk 0 köşe noktalarını belirtir
            plate = [int(i) for i in [minx,maxx,w,h]] # x,y,w,h
            
            plt.title("plate detected")    
            control = True
        else:
            # Plaka tespit edilemedi
            cv2.drawContours(img,[box],0,(0,0,255),2)
            plt.title("plate not detected")
            
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.show()
        
        if (control):
            break

# plaka bulunmuştur

'''


def plateDetection(img):

    img_bgr = img
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Convert image to grayscale

    # Kenar tespiti
    # transaction image ti_img

    # Kenarlıkları daha belirgin hale getirmek için median blur uygulandı
    # Median blur resimi 5x5 bir kernel(boyut) ile filtreler ve her pikselin değerini ortanca değer ile değiştirir

    ti_img = cv2.medianBlur(img_gray,5) # Apply median blur to the image
    ti_img = cv2.medianBlur(ti_img,5) # Apply median blur to the image

    # Yoğunluk merkezi almak için median(ortanca değeri) değeri hesaplandı
    # Kenar tespiti için alt ve üst yoğunluk merkezi belirlendi
    # Canny algoritması ile kenar tespiti yapıldı

    median = np.median(ti_img)

    low = 0.67 * median # 3/2 alt yoğunluk merkezi
    high = 1.33 * median # 4/3 üst yoğunluk merkezi

    border = cv2.Canny(ti_img,low,high) # Apply Canny edge detection to the image

    # Kenar tespiti daha iyi yapılabilmesi için genişletme işlemi yapılması lazım

    border = cv2.dilate(border,np.ones((3,3),np.uint8),iterations=1) # Apply dilation to the image

    # Aynı piksel değerlerine sahip piksellerin bir araya toplanmasına contour denir.
    # Contour tespiti yapıldı

    # RETR_TREE: Tüm hiyerarşik yapıyı döndürür
    # CHAIN_APPROX_SIMPLE: Kenar noktalarını sıkıştırır ve sadece kenar noktalarını döndürür

    cnt = cv2.findContours(border,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # Find contours in the image
    cnt = cnt[0]
    cnt = sorted(cnt,key=cv2.contourArea,reverse=True) # [:10] # Sort the contours by area and get the largest 10 contours

    H,W = 500,500
    plate = None

    for c in cnt:
        rectanial = cv2.minAreaRect(c) # Dikdörtgen yapısını alır
        (x,y),(w,h),r = rectanial # r: döndürme açısı
        if(w>h and w>h*2) or (h>w and h>w*2) : # Plaka boyutu belirlendi
            box = cv2.boxPoints(rectanial) # Dikdörtgenin köşe noktalarını alır # [[0,1],[12,13],[14.30],[25,23]]
            box = np.int64(box) # 64 almamızın sebebi köşe noktalarının bazen resmin dışında kalıp - değer alabiliyor o yüzden
            # tam sayı alırız
            
            minx = np.min(box[:,0])
            miny = np.min(box[:,1])
            maxx = np.max(box[:,0])
            maxy = np.max(box[:,1])
            
            potantial_plate = img_gray[miny:maxy,minx:maxx].copy() # copy dememizin sebebi orjinal resmi bozmamak
            potantial_median = np.median(potantial_plate)
            
            control1 = potantial_median > 84 and potantial_median < 200 # Yoğunluk merkezi kontrolü
            control2 = h < 50 and w < 150 # Sınır kontrolü
            control3 = w < 50 and h < 150 # Sınır kontrolü
            
            print(f"potantial_plate Median: {potantial_median} - Width: {w} - Height: {h}")
            
            control = False
            if (control1 and (control2 or control3)):
                # Plaka tespit edildi
                
                # cv2.drawContours(img,[box],0,(0,255,0),2) # ilk 0 köşe noktalarını belirtir
                plate = [int(i) for i in [minx,miny,w,h]] # x,y,w,h
                control = True
            else:
                # Plaka tespit edilemedi
                # cv2.drawContours(img,[box],0,(0,0,255),2)
                pass
            if (control):
                return plate
    return []
  