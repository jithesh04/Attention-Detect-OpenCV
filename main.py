import cv2
from PIL import Image
from face import Face_detection
import numpy as np
res1,res2 = 0,0
f1,f2 = "loading....","loading...."
def updatedValues1():
    global f1
    return f1
def updatedValues2():
    global f2
    return f2
def iterate_video_frames(video_path):
    global f1,f2
    a1,a2 = 0,0
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Couldn't open the video file.")
        return
    frame_count = 0
    while True:
        if frame_count==5:
            break
        ret, frame = video_capture.read()
        if ret:
            print(".",end=" ")
            xframe = Image.fromarray(np.uint8(frame))
            res1,res2 = Face_detection(xframe)
            if frame_count==0:
                a1,a2 = res1,res2
            else:
                a1 = (a1+res1)//2
                a2 = (a2+res2)//2
            frame_count+=1
        else:
            break
    f1,f2 = a1,a2
# Example usage:
video_path = "static/video1.mp4"
iterate_video_frames(video_path)
#sudo apt update
#sudo apt install -y libgl1-mesa-glx libglib2.0-0
