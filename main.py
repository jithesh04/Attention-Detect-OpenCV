import cv2
from PIL import Image
from face import Face_detection
import numpy as np
res1,res2 = 0,0
def updatedValues1():
    global res1
    return res1
def updatedValues2():
    global res2
    return res2
def iterate_video_frames(video_path):
    global res1,res2
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Couldn't open the video file.")
        return
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if ret:
            xframe = Image.fromarray(np.uint8(frame))
            res1,res2 = Face_detection(xframe)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                b'COntent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Example usage:
video_path = "video1.mp4"
iterate_video_frames(video_path)
