import numpy as np
import sys
import os
import array
import cv2
import parameters as params


def get_frame_count(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("[Error] video={} can not be opened.".format(video_file))
        sys.exit(-6)

    num_frames = int(cap.get(7))
    fps = round(cap.get(5), 2)
    width = cap.get(3)
    height = cap.get(4)
    print("width = ", width)
    print("height = ", height)

    if not fps or fps != fps:
        fps = 30   

    return num_frames, fps