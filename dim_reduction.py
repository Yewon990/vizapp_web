import os 
from flask import Flask, render_template
import pandas as pd
import json
import plotly
import plotly.express as px
import umap
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from parameters import main_parameter as main_params

#app = Flask(__name__)

class DR():     #dimension_reduction
    def __init__(self, options, basic_path, video_name, od_result):
        self.options = options
        self.basic_path = basic_path
        self.video_name = video_name
        self.data_path = main_params.feature_folder + options['feature'] + '/' + video_name
        
        self.od_result = od_result

        self.data = []
        self.y = []
        self.seg =[]
        self.size_list = []
        self.tmp_list=[]
        self.idx=0
        self.point_size = 1

        print('data path: ', self.data_path)
        new_od_result = []
        for seg in self.od_result:
            new_od_result.append('{0:06d}'.format(seg))
        print('check od result', self.od_result)
        print('new', new_od_result)

        npy_list = os.listdir(self.data_path)
        count =0
        for npy_file in npy_list:
            npy_path = self.data_path + '/' + npy_file
            anomaly_idx = npy_file.split('.')[0].split('_')[-1]
            # if anomaly_idx in self.od_result:
            if anomaly_idx in new_od_result:
                print(anomaly_idx)
                count += 1
                self.y.append(self.video_name + '_anomaly')
            else:
                self.y.append(self.video_name + '_normal')
            self.size_list.append(int(self.point_size))
            self.seg.append(self.idx)
            self.idx += 1
            npfile = np.load(npy_path)
            self.data.extend(npfile)

        print(count)
            
        self.tmp_list.append(self.seg)
        self.tmp_list.append(self.y)
        self.tmp_list.append(self.size_list)

        self.data = np.array(self.data)
        self.tmp_list = np.array(self.tmp_list)
        self.tmp_list = np.transpose(self.tmp_list)

        # model implemetation
    def run_DR(self):
        if self.options['reducer'] == 'UMAP':
            model = umap.UMAP(n_components=self.options['n_components'], n_neighbors=self.options['n_neighbors'], min_dist=self.options['min_dist'])
            result = model.fit_transform(self.data)
        elif self.options['reducer'] == 'T-SNE':
            model =TSNE(n_components=self.options['n_components'])
            result = model.fit_transform(self.data)
        elif self.options['reducer'] == 'PCA':
            model = PCA(n_components=options['n_components'])
            result = model.fit_transform(self.data)
        

        print("Dimensionality reduction complete")

        if self.options['n_components'] == 3:
            df = pd.DataFrame(data = result, columns=["in1", "in2", "in3"])
            tmp_df = pd.DataFrame(data = self.tmp_list, columns=["segment", "is_anomaly", "size_ele"])
            df = pd.concat([df, tmp_df], axis=1)
            df["size_ele"] = pd.to_numeric(df["size_ele"], errors='coerce')
            df = df.dropna(subset=['size_ele'])
            df = df.astype({'size_ele':'int'})
            
            fig = px.scatter_3d(df, x="in1", y="in2", z="in3", color="is_anomaly", size="size_ele",
                                hover_data={"segment":True, 
                                            "is_anomaly":True, 
                                            "in1":False,
                                            "in2":False,
                                            "in3":False,
                                            "size_ele":True}
                                    )
        elif self.options['n_components'] == 2:
            df = pd.DataFrame(data = result, columns=["in1", "in2"])
            tmp_df = pd.DataFrame(data = self.tmp_list, columns=["segment", "is_anomaly", "size_ele"])
            df = pd.concat([df, tmp_df], axis=1)
            
            df = df.astype({'size_ele':'int'})

            fig = px.scatter(df, x="in1", y="in2", color="is_anomaly", size="size_ele",
                                hover_data={"segment":True, 
                                            "is_anomaly":True, 
                                            "in1":False,
                                            "in2":False,
                                            "size_ele":False})

        fig.update_layout(clickmode='event+select')
        fig.update_traces(marker_size=10)

        return fig