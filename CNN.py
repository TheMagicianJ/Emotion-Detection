
import numpy as np
import cv2
import os
import mediapipe as mp
import FaceDetection as fd
import HandDetection as hd
import ConvolutionalLayer

FILE_PATH = "archive/videos/00335.mp4"
PADDING = 20
INPUT_WIDTH = 640
INPUT_HEIGHT = 480

cap = cv2.VideoCapture(0)


# Hand Deection

hand_detector = hd.HandDetector()



# Face Detection

face_detector = fd.FaceDetector()



# Drawing Utility

drawlms = mp.solutions.drawing_utils



#Cropping and Resizing

def findPerson(img):
        
        found = False
        
        # Hand Detection
        
        img,hands_list  = hand_detector.findHands(img)

        # Face Detection

        img, face_list = face_detector.findFaces(img)

        # Get the dimensions of the detected person 
        
        person_dimensions = []

        if len(face_list) > 0:
            
            for face in face_list:
                
                 # Get dimensions of the identified face

                face_width = face[1][0] - face[0][0]
                face_height = face[1][1] - face[0][1]

                # Identify a point to focus on for the identified person

                tracking_point = (face[0][0] + int(0.5 * face_width), max((face[0][1] + 2 * face_height),0))
                
                # Creating thw points of the bounding box
                x1 = max((tracking_point[0] - int(2 * face_width)), 0)
                x2 = min((tracking_point[0] + int(2 * face_width)), img.shape[1])
                y1 = max((tracking_point[1] - int(2 * face_height)),0)
                y2 = min((tracking_point[1] + int(2 * face_height)), img.shape[0])


                lower_bound = max((tracking_point[0] - int(2 * face_width)),0), min((tracking_point[1] + int(2.5 * face_height)), img.shape[0])
                upper_bound = min((tracking_point[0] + int(2 * face_width)), img.shape[1]), max((tracking_point[1] - int(2.5 * face_height)),0)
                
                
                # Store the bounbding box and tracking point of each person in a list

                person_dimensions.append([lower_bound, upper_bound, tracking_point])

                print(str(face[1][0] + (5 * face_width)))

            img = cv2.rectangle(img, lower_bound, upper_bound, (255,30,30), 1)
            img = cv2.circle(img,tracking_point,5 ,(30,255,30), 2)
          

           
        # Only decide if there's a person of both the hands and face are detected
        print(person_dimensions)
        if (len(face_list) > 0) and (len(hands_list) > 0):
            
            found = True

            #img = cv2.resize(img,(INPUT_WIDTH, INPUT_HEIGHT))

        #img = cv2.rectangle(img,(min(x_list), min(y_list)),(max(x_list),max(y_list)), (255,30,30), 1 )


        return img, found, face_list, person_dimensions



#To center around the identified person and make sure all the inputs are of the same size

def cropResize(img, dimensions):

    # Crop the image to the persons border
    for person in dimensions:
         
         print("Lower bound: " + str(person[0][0]) + " , " + str(person[0][1]))
         print("Upper bound: " + str(person[1][0]) + " , " + str(person[1][1]))

         

         img = img[person[1][1] : person[0][1], person[0][0] : person[1][0]]

    # Resize the image to input 

    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))



    #x_padding = int(200 * (y[0] - x[0])/ img.shape[0])
    #y_padding = int(200 * (y[1] - x[1])/ img.shape[1]) 
    
    #min_x = max(x[0] - x_padding, 0)
    #max_x = min(y[0] + x_padding, img.shape[0])

    #min_y = max(x[1] - y_padding, 0)
    #max_y = min(y[1] + y_padding, img.shape[1])

    #img = img[min_y : max_y, min_x : max_y]
    
    #img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))

    #print(x_padding)
    #print(y_padding)
    #print("x: " + str(x[0]) + " , " + str(x[1]))
    #print("y: " + str(y[0]) + " , " + str(y[1]))
    

    return img



#This is iterating through each frame of a given video

# Creating the VGG16 Model



while not cap.isOpened():
    cap = cv2.VideoCapture(0)

    cv2.waitKey(1000)
    print("Wait for the header")

#person_seen = False
no_person_seen = 0
count = 0
while True: 

    flag, frame = cap.read()


    if flag:

        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        fHeight, fWidth, fDepth = frame.shape

        Network = [ConvolutionalLayer(frame.shape,(3,3),3),
                   ConvolutionalLayer()]
        
        #if person_seen == False:
        
        #frame, found, faces, person_dimensions = findPerson(frame)
        #person_seen = found

        #if found:

            #frame = cropResize(frame, person_dimensions)
            

        

        count += 1
    else:

        cap.set(cv2.CAP_PROP_POS_FRAMES)
        print("Frame not ready")
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:

        break

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):

        break

    cv2.imshow("Hello, World!", frame)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()  




    


