import os
import matplotlib.pyplot as plt
import cv2

veri = os.listdir("Dataset Review")

for image_url in veri:
    img = cv2.imread("Dataset Review/"+image_url)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(500,500))
    plt.imshow(img)
    plt.show()