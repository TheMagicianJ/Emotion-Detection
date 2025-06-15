import cv2
import os
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FILE_PATH = ""

# Face Detection

base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# To-do: Get rid of FaceDetection module and put it here.

def findFace(img):

      results = detector.detect(img)
      bbox = None

      for detection in results:

            bbox = detection.bounding_box
      
      return bbox  


def crop(img, face):

    # Crop the image to the faces bounding box
    img = img[face[1][1] : face[0][1], face[0][0] : face[1][0]]

    return img


def reSize(img,dimensions):
      
      #Resizing the img to the input size of network

      img = cv2.resize(img, dimensions)
      
      return 

def createFileList(dir):
      
      #Creating a the list of files for where the image files are stored
      fileList = []
      for root,dirs, files in os.walk(dir, topdown = False):
            for name in files:
                  fullName = os.path.join(root, name)

      return fileList



#Pre-processing the data

imgList = createFileList(FILE_PATH)

for image in imgList:
      
      img = mp.Image.create_from_file(image)
      img_copy = np.copy(img.numpy_view())
      boundingBox = findFace(img)
      img = crop(img_copy, boundingBox)

      cv2.imwrite(image,img)

      #TO:Do One hot encoding for emotions.
      # TO:DO Reshape to [depth,width,height]




            
