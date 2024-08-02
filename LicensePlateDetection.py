import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

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