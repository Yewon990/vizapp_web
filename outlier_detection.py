import os
import sys
import random
import glob
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import parameters.main_parameter as params
from pyod.models.iforest import IForest
from pyod.models.cblof import CBLOF
# from OD_models.LOF import LOF

class OD():
    def __init__(self, test_video, options, exist_trainset):
        self.basic_path = params.basic_path
        self.data_folder = params.folder #anomaly class *kidnap
        self.data_path = self.basic_path + self.data_folder
        self.od_models = options['odmodels']
        self.feature = options['feature']
        self.test_video = test_video
        self.n_cluster_num = params.n_cluster_start
        self.exist_trainset = exist_trainset

    def outlier_detection(self):
        if self.od_models == 'LOF':
            clf_name = 'CBLOF'
            clf = CBLOF(random_state=777, n_jobs=-1, use_weights=params.use_weights, n_clusters=self.n_cluster_num)
        elif self.od_models == 'iforest':
            clf_name = 'IForest'
            clf = IForest()

        if self.exist_trainset:
            anomaly_seg_num = []
            mp4_list = os.listdir(self.data_path)
            print("data path: ", self.data_path)
            print("test video: ", self.test_video)

            inlier = []
            outlier = []

            for mp4 in mp4_list:  
                if mp4 == self.test_video:
                    continue
            
                mp4_path = self.data_path + '/' + mp4
                normal_npy_path = glob.glob(mp4_path + '/*normal.npy')
                anomaly_npy_path = glob.glob(mp4_path + '/*anomaly.npy')

                for normal_npy in normal_npy_path:
                    data = np.load(normal_npy)
                    inlier.extend(data)
                    
                for anomaly_npy in anomaly_npy_path:
                        data = np.load(anomaly_npy)
                        outlier.extend(data)

            print('load numpy done')
            random.seed(params.outlier_seed)
            assert len(inlier) * params.outlier_percent <= len(outlier)
            outlier = random.sample(outlier, int(len(inlier) * params.outlier_percent))  #랜덤 샘플 outlier 비율 0.1

            inlier = np.array(inlier)
            outlier = np.array(outlier)

            inlier_train = inlier
            outlier_train = outlier

            print('inlier train: ', inlier_train.shape)
            print('outlier train: ', outlier_train.shape)

            y_train = []

            for i in range(len(inlier_train)):
                y_train.append(0)
            
            for i in range(len(outlier_train)):
                y_train.append(1)

            inlier_train = inlier_train.tolist()
            outlier_train = outlier_train.tolist()

            X_train = inlier_train
            X_train.extend(outlier_train)
        
            test_npy_path = glob.glob(params.feature_folder + self.feature + '/' + self.test_video + '/*.npy')
            test_npy_path.sort()

            y_test =[]

            for i in range(len(test_npy_path)):
                if 'anomaly' in test_npy_path[i]:
                    y_test.append(1)
                else:
                    y_test.append(0)

            X_test = []
            for npy in test_npy_path:
                data = np.load(npy)
                X_test.extend(data)

            # try:
            clf.fit(X_train)

            y_train_pred = clf.labels_
            y_train_scores = clf.decision_scores_

            y_test_pred = clf.predict(np.array(X_test, dtype = float))
            y_test_scores = clf.decision_function(np.array(X_test, dtype = float))
            # except Exception as e:
                # print(e)

        else:
            X_test = []            
            test_npy_path = glob.glob(params.feature_folder + self.feature + '/' + self.test_video + '/*.npy')
            test_npy_path.sort()

            for npy in test_npy_path:
                data = np.load(npy)
                X_test.extend(data)

            y_test_pred = clf.fit_predict(np.array(X_test, dtype = float))

        seg_num = 0
        anomaly_seg_list = []
        od_result = []
        for i in y_test_pred:
            seg_num += 1
            if i == 1:
                anomaly_seg_list.append(seg_num)
        
        return anomaly_seg_list