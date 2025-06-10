import numpy as np
import cv2
import os
import mediapipe as mp
import FaceDetection as fd


FILE_PATH = ""

# Face Detection

face_detector = fd.FaceDetector()

# To-do: Get rid of FaceDetection module and put it here.

def findFace(img):
        
        # Face Detection

        img, face = face_detector.findFace(img)

        return img, face


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
                  fileList.append(fullName)

      return fileList





#Pre-processing the data

imgList = createFileList(FILE_PATH)

for image in imgList:
      
      img = cv2.imread(image)
      boundingBox = findFace(img)
      img = crop(img, boundingBox)
      cv2.imwrite(image,img)

      #Convert to npArray?
      #One hot encoding?





           

            
