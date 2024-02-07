import cv2
import dlib
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg 


from HistoryDict import *



class EyesTracking():

    def __init__(self):
    
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.webcam.set(cv2.CAP_PROP_FPS, 60)

        self.leftEyeHistory  = History(1, 20)
        self.rightEyeHistory = History(1, 20)

        cv2.namedWindow('image')

        def nothing(x):
            pass
        cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

    def print_eye_pos(self, img, left, right):
        """
        Print the side where eye is looking and display on image
        Parameters
        ----------
        img : Array of uint8
            Image to display on
        left : int
            Position obtained of left eye.
        right : int
            Position obtained of right eye.
        Returns
        -------
        None.
        """
        if left == right and left != 0:
            text = ''
            if left == 1:
                print('Looking left')
                text = 'Looking left'
            elif left == 2:
                print('Looking right')
                text = 'Looking right'
            elif left == 3:
                print('Looking up')
                text = 'Looking up'
            font = cv2.FONT_HERSHEY_SIMPLEX 
            cv2.putText(img, text, (30, 30), font,  1, (0, 255, 255), 1, cv2.LINE_AA) 

    def shape_to_np(self, shape, dtype = "int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype = dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

    # 建立眼球遮罩
    def eye_on_mask(self, face_shape, mask, side):

        points = [face_shape[i] for i in side]

        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)

        l = points[0][0]
        t = (points[1][1]+points[2][1])//2
        r = points[3][0]
        b = (points[4][1]+points[5][1])//2

        return mask, [l, t, r, b]

    def find_eyeball_position(self, end_points, cx, cy):
        x_ratio = (end_points[0] - cx)/(cx - end_points[2])
        y_ratio = (cy - end_points[1])/(end_points[3] - cy)
        if x_ratio > 3:
            return 1
        elif x_ratio < 0.33:
            return 2
        elif y_ratio < 0.33:
            return 3
        else:
            return 0

    def contouring(self, thresh, mid, end_points, img, right=False):
        _, cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

        try:
            cnt = max(cnts, key = cv2.contourArea)
            M = cv2.moments(cnt)

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if right:
                cx += mid

            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 1)

            cv2.circle(img, (end_points[0], end_points[1]), 2, (0, 255, 0), 1)
            cv2.circle(img, (end_points[2], end_points[3]), 2, (255, 0, 0), 1)


            pos = self.find_eyeball_position(end_points, cx, cy)
            return pos
        except:
            pass

    def run(self):
        kernel = np.ones((5, 5), np.uint8)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
        LEFT_EYE_EIGEN_POINT    = [36, 37, 38, 39, 40, 41]
        RIGHT_EYE_EIGEN_POINT   = [42, 43, 44, 45, 46, 47]
        
        ret, img = self.webcam.read()
        thresh = img.copy()

        while(True):

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, img = self.webcam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = self.shape_to_np(shape)

                mask = np.zeros( img.shape[:2], dtype = np.uint8 )
                mask, end_point_left    = self.eye_on_mask(shape, mask, LEFT_EYE_EIGEN_POINT )
                mask, end_point_right   = self.eye_on_mask(shape, mask, RIGHT_EYE_EIGEN_POINT )
                mask = cv2.dilate( mask, kernel, 5) 
                eyes = cv2.bitwise_and(img, img, mask = mask) 
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = (shape[42][0] + shape[39][0]) // 2
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

                self.leftEyeHistory.add( [end_point_left] )
                if( self.leftEyeHistory.is_full() ):
                    [end_point_left] = self.leftEyeHistory.get_average()
                    self.leftEyeHistory.pop()

                self.rightEyeHistory.add( [end_point_right] )
                if( self.rightEyeHistory.is_full() ):
                    [end_point_right] = self.rightEyeHistory.get_average()
                    self.rightEyeHistory.pop()

    
                threshold = cv2.getTrackbarPos('threshold', 'image')
                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY) 
                thresh = cv2.erode(thresh, None, iterations=2)  
                thresh = cv2.dilate(thresh, None, iterations=4) 
                thresh = cv2.medianBlur(thresh, 3)  
                thresh = cv2.bitwise_not(thresh)   

                
                eyeball_pos_left    = self.contouring(thresh[:, 0:mid], mid, end_point_left , img, False)
                eyeball_pos_right   = self.contouring(thresh[:, mid:] , mid, end_point_right, img, True)
                self.print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)


            cv2.imshow('eyes', img)
            cv2.imshow("image", thresh)

    def quit(self):
        self.webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    eyesTracking = EyesTracking()
    eyesTracking.run()
    pass