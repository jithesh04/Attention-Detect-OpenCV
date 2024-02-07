import dlib as DLIB
import cv2 as CV2
import numpy as NP


import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from mpl_toolkits.mplot3d import Axes3D  



def shape_to_np(shape, dtype = "int"):
	coords = NP.zeros((68, 2), dtype = dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


def face_detect():
    detector = DLIB.get_frontal_face_detector()
    predictor = DLIB.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

    img = CV2.imread('image.jpg')

    CV2.imshow('image', img)

    rects = detector( img )

    while True:
        for rect in rects:
            shape = predictor(img, rect)
            for i in range(0, 68):
                CV2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), -1)

            shape = shape_to_np(shape)

            image_points = NP.array([
                                    shape[30],   
                                    shape[8],    
                                    shape[36],     
                                    shape[45],     
                                    shape[48],   
                                    shape[54]    
                                ], 
                                dtype='double')
            for element in image_points:
                CV2.circle(img, (int(element[0]), int(element[1])), 3, (0,255,0), -1)

            CV2.imshow('face', img)

        if CV2.waitKey(1) & 0xFF == ord('q'):
            break

    CV2.destroyAllWindows()

face_detect()

def face_model():
    FACE_3D_MODEL_POINT = NP.array([
                                (0.0, 0.0, 0.0),         
                                (0.0, -330.0, -65.0),
                                (-225.0, 170.0, -135.0),     
                                (225.0, 170.0, -135.0),      
                                (-150.0, -150.0, -125.0),    
                                (150.0, -150.0, -125.0)      
                            ], dtype='double')
    FACE_3D_NANE = ['Nose tip', 'Chin', 'Left eye left corner', 'Right eye right corne', 'Left Mouth corner', 'Right mouth corner']
    count = 0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for element in FACE_3D_MODEL_POINT:
        ax.scatter(element[0], element[1], element[2], marker='o')
        ax.text(element[0], element[1], element[2], FACE_3D_NANE[count])
        count += 1

    ax.plot([0.0, 0.0], [0.0, -330], [0.0, -65.0])
    ax.plot([-225.0, 225.0], [170, 170], [-135.0, -135.0])
    ax.plot([-225.0, -150.0,], [170, -150.0,], [-135.0, -125.0])
    ax.plot([225.0, 150.0,], [170, -150.0,], [-135.0, -125.0])
    ax.plot([-150.0, 150.0,], [-150, -150.0,], [-125.0, -125.0])
    ax.plot([-150.0, 0.0,], [-150, -330.0,], [-125.0, -65.0])
    ax.plot([150.0, 0.0,], [-150, -330.0,], [-125.0, -65.0])
    ax.plot([-225.0, 0.0], [170.0, 0.0], [-135.0, 0.0])
    ax.plot([225.0, 0.0], [170.0, 0.0], [-135.0, 0.0])

    plt.show()