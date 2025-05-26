import cv2
import mediapipe as mp

class HandDetector:

    def __init__(self, mode = False, maxHands = 2,  minDetectCon = 0.8, minTrackCon = 0.8):

        self.mode = mode
        self.maxHands = maxHands
        self.minDetectCon = minDetectCon
        self.minTrackCon = minTrackCon

        self.mpHandDetection = mp.solutions.hands
        self.handDetection = self.mpHandDetection.Hands()#self.mode, self.maxHands,
                                                         #self.minDetectCon,self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils



    def findHands(self, img):

        self.results = self.handDetection.process(img)
        h,w,c = img.shape

        hand_list = []

        if self.results.multi_hand_landmarks:

            

            for hand_lms in self.results.multi_hand_landmarks:

                xlist = []
                ylist = []
                
                for lm in hand_lms.landmark:
                    
                    px, py = int(lm.x * w), int(lm.y * h) # x and y are given as frations of the frame to get to pixels must multiply by the frames width and height

                    xlist.append(px)
                    ylist.append(py)
                    # Creating Bounding box

                    #if px > highx:
                    #    highx = px

                    #if py > highy:
                    #    highy = py

                    #if px < lowx or lowx == 0:
                    #    lowx = px

                    #if py < lowy or lowy == 0:
                    #    lowy = py

                    #Creating Bounding boxes
                 
                #Sort the list of coordinates to get the
                #sortedx = sorted(xlist)
                #sortedy = sorted(ylist)

                #lowx = sortedx[0]
                #lowy = sortedy[0]

                #highx = sortedx [len(xlist) - 1]
                #highy = sortedy[(len(ylist) - 1)]


                
                bottom_corner = int(min(xlist)), int(min(ylist))
                top_corner = int(max(xlist)), int(max(ylist))
                hand_list.append([bottom_corner,top_corner])

 

                cv2.rectangle(img, bottom_corner, top_corner, (255,30,30), 1)
                self.mpDraw.draw_landmarks(img, hand_lms)

        return img, hand_list



def main():

    FILE_PATH = "archive/videos/00335.mp4"
    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while not cap.isOpened():

        cap = cv2.VideoCapture(0)
        cv2.waitKey(1000)
        print("Wait for the header")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    count = 0
    
    while True: 

        flag, frame = cap.read()
        
        if flag:

            frame, hands_list = detector.findHands(frame)
            print(str(pos_frame) + " frames")

            if len(hands_list) > 0:
                print("Bounding box 1 at " + str(hands_list[0][0]) + str(hands_list[0][1]))
            if len(hands_list) > 1:
                print("Bound box 2 at " + str(hands_list[1][0]) + str(hands_list[1][1]))

            print(len(hands_list))


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
