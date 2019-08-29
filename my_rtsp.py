# -*- coding: utf-8 -*-
import cv2
import time
import threading
import sys
import argparse
import numpy as np
import colorsys
from yolo import YOLO
from PIL import Image

from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model


RTSP_URL = "rtsp://192.168.1.125/unicast/video:1"
UVC_URL = "/dev/video0"


class MyCamera:
    def __init__(self, URL):
        self.frame = []
        self.status = False
        self.is_stop = False
        self.capture = cv2.VideoCapture(URL)

    def start(self):
        print('MyCamera started!')
        threading.Thread(target=self.read_frame, daemon=True, args=()).start()

    def stop(self):
        self.is_stop = True
        print('MyCamera stopped!')

    def get_frame(self):
        return self.frame

    def get_status(self):
        return self.status

    def read_frame(self):
        while (not self.is_stop):
            self.status, tmp = self.capture.read()
            if (self.status == True):
                self.frame = tmp

        self.capture.release()



def open_webcam_1(yolo, video_path, output_path=""):
    print("open_webcam_1")

    webcam = MyCamera(video_path)
    webcam.start()
    time.sleep(1)

    try:
        while True:

            if webcam.get_status() == False:
                raise IndexError('open camera failed! open again...')

            frame = webcam.get_frame()

            # gray
            #frame = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            #frame = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

            frame = Image.fromarray(frame)
            image = yolo.detect_image(frame)
            result = np.asarray(image)
            cv2.imshow('Image1', result)

            if cv2.waitKey(10) == 27:
                cv2.destroyAllWindows()
                webcam.stop()
                break
            '''
            if cv2.waitKey(1000) == ord('p'):
                print("snapshot!")
                threading.Thread(target=save_picture_1, daemon=True, args=(I,)).start()
            '''

    except :
        print("an exception occurred")
        webcam.stop()
        time.sleep(1)
        open_webcam_1(yolo, video_path, output_path)

    webcam.stop()
    cv2.destroyAllWindows()


def open_webcam_2(video_path):
    print("open_webcam_2")
    webcam = cv2.VideoCapture(video_path)
    ret, frame = webcam.read()
    try:
        while ret:
            ret, frame = webcam.read()
            cv2.imshow("Image2", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except:
        print("an exception occurred")
        open_webcam_2()

    cv2.destroyAllWindows()
    webcam.release()


def open_picture_1():
    print("open_picture_1")
    image = cv2.imread("/home/bu10/Pictures/cat.jpeg")
    cv2.imshow("pic1", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def open_picture_2():
    print("open_picture_2")
    image = cv2.imread("/home/bu10/Pictures/car.jpg")
    cv2.imshow("pic2", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def save_video_1(video_path=RTSP_URL):
    print("save_video_1")

    webcam = MyCamera(video_path)
    webcam.start()
    time.sleep(1)

    f = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('a.mp4', f, 16.0, (1280, 720))

    try:
        while True:

            if webcam.get_status() == False:
                raise IndexError('open camera failed! open again...')


            frame = webcam.get_frame()
            if webcam.get_status() == True:
                out.write(frame)
            else:
                continue



            cv2.imshow('Image1', frame)

            if cv2.waitKey(10) == 27:
                cv2.destroyAllWindows()
                webcam.stop()
                break

    except :
        print("an exception occurred")
        webcam.stop()
        time.sleep(1)
        out.release()
        save_video_1(video_path)

    finally:
        out.release()
        webcam.stop()
        cv2.destroyAllWindows()


def save_picture_1(frame):
    print("save_picture_1")
    cv2.imwrite('a.jpg', frame)


def system_pause():
    while True:
        time.sleep(3)


def main():

    #Save video
    #threading.Thread(target=save_video_1, daemon=True, args=()).start()

    #UVC
    #threading.Thread(target=open_webcam_2, daemon=True, args=(UVC_URL, )).start()

    #Show picture
    #threading.Thread(target=open_picture_1, daemon=True, args=()).start()
    #threading.Thread(target=open_picture_2, daemon=True, args=()).start()




    #class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()
    #open_webcam_1(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    system_pause()



if __name__ == '__main__':
    main()



__author__ = "Will Nien"
__email__ = "will.nien@quantatw.com"
__version__ = "1.0.0"
