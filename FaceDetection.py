import cv2
import mediapipe as mp

class FaceDetector():

    def __init__(self, minDetectCon = 0.8):

        self.minDetectCon = minDetectCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.75)


    def findFace(self, img):

        self.results = self.faceDetection.process(img)

        face_top_corner = 0
        face_bottom_corner = 0

        if self.results.detections:
            
            for detection in self.results.detections:

                fbboxC = detection.location_data.relative_bounding_box
                h,w,c = img.shape

                face_bottom_corner = int(fbboxC.xmin * w),int(fbboxC.ymin * h)
                face_top_corner = int((fbboxC.xmin + fbboxC.width) * w), int((fbboxC.ymin + fbboxC.height) * h)
                face = (face_bottom_corner, face_top_corner)
                 
                cv2.rectangle(img, face_bottom_corner, face_top_corner, (255,30,30), 1)

        return img, face

       
def main():

    FILE_PATH = "archive/videos/00335.mp4"
    cap = cv2.VideoCapture(0)

    detector = FaceDetector()

    while not cap.isOpened():
        cap = cv2.VideoCapture(0)

        cv2.waitKey(1000)
        print("Wait for the header")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    count = 0
    
    while True: 
        flag, frame = cap.read()
        
        if flag:

            frame, faces = detector.findFaces(frame)

            print(str(pos_frame) + " frames")
            if len(faces) > 0:
                
                print("Bounding box between: " + str(faces[0][0]) + ", " + str(faces[0][1]))
            
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



if __name__ == "__main__":
    main()