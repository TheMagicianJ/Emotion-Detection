import cv2
import os
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import shutil


IMAGES_PATH = "C:/Users/james/Documents/Masters/Dissertation/Dataset/val_set/images"
PROCESSED_PATH = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Test/Test_images"

DATA_PATH = "C:/Users/james/Documents/Masters/Dissertation/ExpectedTest"
TARGET_DATA_PATH = "C:/Users/james/Documents/Masters/Dissertation/Preprocessed_Test/Test_expected"

# Face Detection

# TO:DO - MediaPipe >:
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# To-do: Get rid of FaceDetection module and put it here.

def findFace(img: np.ndarray) -> tuple:

      found = False

      results = detector.detect(img)
      bbox = ()

      for detection in results.detections:
            
            bbox = detection.bounding_box

      return bbox  


def crop(img: np.ndarray, face) -> np.ndarray:

    # Crop the image to the faces bounding box
    img = img[face.origin_x : face.origin_x+ face.width, face.origin_y : face.origin_y + face.height]

    return img


def createFileList(dir: str) -> list:
      
      #Creating a the list of files for where the image files are stored
      fileList = []
      
      for root,dirs, files in os.walk(dir, topdown = False):


            for file in files:

                  fileList.append(file)

      fileList.sort()

      return fileList


def getExpected(dir: str, target: str, size: int = 10000):

      for r,d,f in os.walk(PROCESSED_PATH, topdown= False):

            for root,dirs, files in os.walk(dir, topdown = False):

                  for file in files:

                        print(file)
                   
                        if "exp" in file and (file.strip("_exp.npy") + ".jpg") in f:

                              shutil.copyfile(f"{dir}/{file}", f"{target}/{file}")

                        


      


#Pre-processing the data

# imgList = createFileList(FILE_PATH)

# for image in imgList:
      
#       img = mp.Image.create_from_file(image)
#       boundingBox = findFace(img)
#       img_copy = np.copy(img.numpy_view())
#       img = crop(img_copy, boundingBox)


#       # transform = transforms.ToTensor
#       # img = transform(img)
#       # img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

#       # Resize the input image to be 3 dimensionall 244 x 244 array
#       img = np.resize(img,(3,244,244))
#       cv2.imwrite(image,img)

def preprocess(dir: str , target: str, size: int = 10000):

      

      for root,directs,files in os.walk(dir, topdown=False):
            
            if size > len(files):

                  print("Size exceeds number of images. Proceeding with max number of images.")

                  size = len(files)

            i = 1
            filesSaved = 0

            for file in files:
                  
                  img = mp.Image.create_from_file(f"{dir}/{file}")
                  boundingBox = findFace(img)

                  if boundingBox == ():

                        print("Face not detected. Skipping Image.")
                        continue

                  for r,d,expf in os.walk(DATA_PATH):

                        for expfile in expf:

                              con = (file.strip(".jpg") + "_exp.npy") == expfile

                              if con == True:

                                    shutil.copyfile(f"{DATA_PATH}/{expfile}", f"{TARGET_DATA_PATH}/{i}.npy")
                                    print(f"Expected Value for {file} is {expfile} and has been saved")
                                    break


                  img = img.numpy_view()
                  np_img = np.copy(img)
                  # img = np.array(img)
                  img = crop(np_img, boundingBox)

                  img = Image.fromarray(img)

                  filename = (f"{target}/{i}.jpg")

                  img = img.save(filename)

                  filesSaved += 1

                  print(f"Image {i} out of {size} saved.")
                  
                  if filesSaved == size:

                        print(f"{size} Images fully saved")
                        break


                  

                  i += 1

                  
                        
                  




#print(list)

def getExpects(path):

      zipped = os.walk(path).__next__()


      for f in zipped[2]:

            if "exp" in f:

                  shutil.copyfile(f"{path}/{f}",f"{TARGET_DATA_PATH}/{f}")
                  print(f" File {f} saved to {TARGET_DATA_PATH}")

            

preprocess(IMAGES_PATH, PROCESSED_PATH, 100000000)
#getExpected(DATA_PATH, TARGET_DATA_PATH)

#getExpects(DATA_PATH)




      #TO:Do One hot encoding for emotions.
      # TO:DO Reshape to [depth,width,height]




            