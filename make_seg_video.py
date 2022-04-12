import cv2
import numpy as np
import glob
import os
import time
import parameters.main_parameter as main_params

class Make_Video():
    # Todo: parameter로 뺄 수 있는 변수 빼보기 (video_path, feature_type, segment_path)
    def __init__(self, seg_video_path):
        self.seg_video_path = seg_video_path        #saved segment path
        self.video_path = main_params.video_folder
        self.segment_path = main_params.segment_folder

    def make_seg_video(self, segment_list, video_name, feature_type):
        img_array = []

        print("=============print video name==============")
        print(video_name)
        print("=====================print segment_list====================")
        print("segment_list :", segment_list)

        if feature_type == "I3D" or feature_type == "SlowFast":
            filename = glob.glob(self.segment_path + 'seg_64/' + video_name + '/*')
        else:
            filename = glob.glob(self.segment_path + 'seg_16/' + video_name + '/*')

        filename.sort()
        segment_list.sort()

        if os.path.exists(self.seg_video_path) == False:
            os.mkdir(self.seg_video_path)

        save_path = self.seg_video_path + video_name + '/'
        print("save_path: ", save_path)

        #get video info
        #video = cv2.VideoCapture(self.video_path + video_name + '.mp4')
        #fps = video.get(cv2.CAP_PROP_FPS)
        fps = 30
        size = (398, 224)       # Todo: frame에서 size 가져오기?
        # height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        # size = (height, width)
        # print("===========H W=====================")
        # print(height, width)

        for files in filename:
            print(files)
            tmp = files.split('_')[-1]
            print("====================tmp===================")
            print(tmp)
            seg = int(tmp)
            seg_str = str(seg)
            print("seg:", seg)

            if seg_str in segment_list:
                frames = glob.glob(files + '/*')
                frames.sort()
                print("in if seg:", seg)

                print(files)
                for frame in frames:
                    img = cv2.imread(frame)
                    height, width, layers = img.shape
                    size = (width, height)
                    img_array.append(img)

        if os.path.exists(save_path) == False:
            os.mkdir(save_path)

        #video_count = len(os.listdir(save_path)) + 1
        make_video_path = save_path + video_name + 'seg_video' + '(' + time.strftime('%c', time.localtime(time.time())) + ')' + '.mp4'
        result_video = cv2.VideoWriter(make_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        #result_video = cv2.VideoWriter(make_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        for i in range(len(img_array)):
            result_video.write(img_array[i])

        result_video.release()
        print("video release!")

        return make_video_path