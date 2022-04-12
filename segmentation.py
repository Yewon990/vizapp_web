import numpy as np
import sys
import os
import subprocess
import array
import cv2
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import parameters.main_parameter as main_params

def get_frame_count(video):
    ''' Get frame counts and FPS for a video '''
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("[Error] video={} can not be opened.".format(video))
        sys.exit(-6)

    num_frames = int(cap.get(7))
    fps = cap.get(5)
    width = cap.get(3)
    height = cap.get(4)
    print("width = ", width)
    print("height = ", height)

    if not fps or fps != fps:
        fps = 30   

    return num_frames, fps

def extract_frames(video, start_frame, frame_dir, num_frames_per_clip, count, video_id):
    ''' Extract frames from a video using opencv '''

    if os.path.isdir(frame_dir):
        pass
    else:
        os.makedirs(frame_dir)

    # get number of frames
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print ("[Error] video={} can not be opened.".format(video))
        sys.exit(-6)
    
    # move to start_frame
    cap.set(1, start_frame)
    
    clip = []
    # grap each frame and save
    for frame_count in range(0, num_frames_per_clip):
        frame_num = frame_count + start_frame

        ret, frame = cap.read()
        
        if not ret:
            print ("[Error] Frame extraction was not successful")
            sys.exit(-7)

        frame_file = frame_dir+ video_id + '_' + str(count) + '_' + '{0:06d}.jpg'.format(frame_num)
        frame = cv2.resize(frame, (224,224))
        clip.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.imwrite(frame_file, frame)


def segmentation(video_file, net):
    if net == 'I3D' or net == 'SlowFast':
        num_frames_per_clip = 64
    else: num_frames_per_clip=16

    # sampling rate (in seconds)
    sample_every_N_sec = 1
    segment = 0
    # don't extract beyond this (in seconds)
    max_processing_sec = 599

    num_frames, fps = get_frame_count(video_file)
    print ("num_frames={}, fps={}".format(num_frames, fps))
    if num_frames < int(sample_every_N_sec * fps):
        start_frame = (num_frames - num_frames_per_clip) / 2
        start_frames = [start_frame]
    else:
        frame_inc = int(sample_every_N_sec * fps)
        frame_inc = num_frames//num_frames_per_clip
        start_frame = int(frame_inc / 2)
        start_frame = 1
        # make sure not to reach the edge of the video
        end_frame = min(num_frames, int(max_processing_sec * fps)) - \
                    num_frames_per_clip
        start_frames = []
        for frame_index in range(start_frame, end_frame, num_frames_per_clip):
            start_frames.append(frame_index)
            segment += 1

    print("segment: ", segment)

    video_id, video_ext = os.path.splitext(
            os.path.basename(video_file)
            )    

    save_dir =  main_params.segment_folder + 'seg_' + str(num_frames_per_clip) + '/' + video_id + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)   
    
    print("segment save path: ", save_dir)
    count = 0  

    for start_frame in start_frames:
        frame_dir = save_dir + video_id + '_' + '{0:06d}'.format(count) + '/'
        clip = extract_frames(video_file, start_frame, frame_dir, num_frames_per_clip, count, video_id)
        count += 1

    return count
