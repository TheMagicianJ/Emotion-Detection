import numpy as np
import cv2
import os


FILE_PATH = "TestImage.jpeg"
PADDING = 20
INPUT_WIDTH = 640
INPUT_HEIGHT = 480


img = cv2.imread("TestImage.jpeg")


#r_img = img[:,:,0]
#g_img = img[:,:,1]
#b_img = img[:,:,2]

#merge_img = cv2.merge([r_img,g_img,b_img])

#print(img.shape)

#cv2.imshow("New Image", merge_img)
#cv2.imshow("Red", r_img)
#cv2.imshow("Blue", b_img)
#cv2.imshow("Green", g_img)

#print(r_img[100:200,100:200].max())
#cv2.waitKey(0)
#cv2.destroyAllWindows

x = 0

for i in range(2):
    print(i)
 