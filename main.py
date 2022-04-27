import os
import plotly.express as px
import pandas as pd
import json
import plotly
import umap
import numpy as np

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from flask import Flask, redirect, render_template, request, send_from_directory, url_for

from tabnanny import filename_only
from werkzeug.utils import secure_filename
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from outlier_detection import OD
from dim_reduction import DR
from edit_json import Edit_Json
from make_seg_video import Make_Video
import get_frame_count

import feature
import cv2
import segmentation as seg
import feature as features
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import parameters.main_parameter as main_params

UPLOAD_FOLDER = main_params.video_folder
FEATURE_FOLDER = main_params.feature_folder
SEGMENT_FOLDER = main_params.segment_folder
MAKE_VIDEO_FOLDER = main_params.make_video_folder
ALLOWED_EXTENSIONS = set(['mp4'])
seg_num = 0

feature_list = ['I3D', 'X3D', 'slowfast']
reducer_list = ['UMAP', 'T-SNE', 'PCA']
od_list = ['iforest','LOF', 'None']
vis_dim_list = [3, 2]

select_segment_list = []
click_segment = 0
seg_btn_click = 0

basic_path = main_params.final_dir

server = Flask(__name__)
server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
server.config['FEATURE_FOLDER'] = FEATURE_FOLDER
server.config['SEGMENT_FOLDER'] = SEGMENT_FOLDER
fig=None

app = Dash(requests_pathname_prefix='/plotly/')

# app.layout = html.Div("initial Dash")
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

edit_json = Edit_Json()
make_video = Make_Video(MAKE_VIDEO_FOLDER)

@server.route('/', methods=['GET','POST'])
def home():
    return render_template('home.html')

def segmentation(filename, feature):
    print("=-=-=-=-=-=-=-=-=-=-=-= segmentation start =-=-=-=-=-=-=-=-=-=-=-=")
    video_file = UPLOAD_FOLDER + filename
    seg_num = seg.segmentation(video_file, feature)
    print("=-=-=-=-=-=-=-=-=-=-=-= segmentation finish=-=-=-=-=-=-=-=-=-=-=-= ")
def feature_extraction(filename, feature):
    print("=-=-=-=-=-=-=-=-=-=-=-= feature extraction start =-=-=-=-=-=-=-=-=-=-=-=")
    features.FeatureExtract(filename, feature)
    print("=-=-=-=-=-=-=-=-=-=-=-= feature extraction end =-=-=-=-=-=-=-=-=-=-=-=")
def hashing_filename(filename):
    old_video_file = UPLOAD_FOLDER+filename
    num_frames, fps = get_frame_count.get_frame_count(old_video_file)
    new_video_file = UPLOAD_FOLDER+filename.split('.')[0] + '(num_frames:' + str(num_frames) + ', fps:' + str(fps) +')' + '.mp4'
    os.rename(old_video_file, new_video_file)
    return new_video_file

@server.route('/visualize', methods=['GET','POST'])
def visualize_temp():
    global video_name, video_name, anomaly_seg_list, fig
    global options
    #options: feature, reducer, odmodels, n_components, n_neighbors, min_dist
    if request.method == 'POST':
        try:
            feature = request.form['feature']
            reducer = request.form['reducer']
            n_neighbors = request.form['n_neighbors']
            min_dist = request.form['min_dist']
            odmodels = request.form['odmodels']
            n_components = request.form['vis_dim']
            n_components = int(n_components)
            n_neighbors = int(n_neighbors)
            min_dist = float(min_dist)
            options = {'feature':feature, 'reducer': reducer, 'odmodels': odmodels, 'n_components': n_components, 'n_neighbors': n_neighbors, 'min_dist': min_dist}
        except Exception as e:
            print(e)
        # options['n_components'] = n_components
        # options['n_neighbors'] = n_neighbors
        # options['min_dist'] = min_dist
        print('all options', options)
        # file 을 list 로 받아옴
        file_list = request.files.getlist('file[]')
        # for file in file_list:
        if file_list[0] and allowed_file(file_list[0].filename):
            video_name = run_extraction(file_list)
            # visualize(video_name, feature, reducer, n_components, n_neighbors, min_dist)
            fig = visualize(video_name, options)
            app.layout = html.Div([
                dcc.Graph(
                    id='visualize_graph',
                    figure=fig
                    ),
                    html.Div([
                        dcc.Markdown("""
                            **Selection Data**
                        """),
                        html.P(id='selected_data'),
                    ], className='three columns'),
                    html.Div(
                        html.Button('Make Video', id='make_video_btn', n_clicks=0)
                    ),
                    html.Div(
                        id = 'btn_test_div', 
                        children='click button?'
                    )
                ])
            return redirect('/plotly/')
    return render_template('visualize_temp.html', feature_list=feature_list, reducer_list=reducer_list, od_list=od_list, vis_dim_list=vis_dim_list)

@server.route('/feature', methods=['GET','POST'])
def only_feature():
    if request.method == 'POST':
        try:
            feature = request.form['feature']
        except Exception as e:
            print(e)
        file_list = request.files.getlist('file[]')
        # for file in file_list:
        if file_list[0] and allowed_file(file_list[0].filename):
            video_name = run_extraction(file_list)
    return render_template('feature.html', feature_list=feature_list)

def run_extraction(file_list):
    filename = secure_filename(file_list[0].filename)
    print('file name', filename)
    video_name = filename.split('.mp4')[0]
    print('video_name', video_name)    
    
    dir_list = os.listdir(server.config['UPLOAD_FOLDER'])
    is_exist = False
    for string in dir_list:
        if video_name in string:
            is_exist = True
        
    # video, feature, segment 있는지 모두 확인
    if is_exist == False:
        try:
            file_list[0].save(os.path.join(server.config['UPLOAD_FOLDER'], filename))
            filename = hashing_filename(filename)
            print('new file name', filename)
            video_name = filename.split('.mp4')[0]
            print('video_name', video_name)
        except Exception as e:
            print(e)
    if options['feature'] == 'I3D' or 'SlowFast':
        if os.path.exists(SEGMENT_FOLDER + 'seg_64/' + video_name) == False:
            try:
                segmentation(filename, options['feature'])
                feature_extraction(video_name, options['feature'])
            except Exception as e:
                print(e)
    elif options['feature'] == 'C3D' or 'X3D':
        if os.path.exists(SEGMENT_FOLDER + 'seg_16/' + video_name) == False:
            try:
                segmentation(filename, options['feature'])
                feature_extraction(video_name, options['feature'])
            except Exception as e:
                print(e)
    print('=-=-=-=-=-=-=-=-=-=-=-= feature extraction & segmentation done =-=-=-=-=-=-=-=-=-=-=-=')
    return video_name

# @server.route('/visualize')
def visualize(video_name, options):
    exist_folder = False
    anomaly_seg_num = []
    if options['odmodels'] == 'None':
        anomaly_seg_num = range(0,seg_num)

    else:
        for section in main_params.folder_list:
            if section in video_name: 
                exist_folder = True
                break
        print('=-=-=-=-=-=-=-=-=-=-=-= Outlier Detection Start =-=-=-=-=-=-=-=-=-=-=-=')
        
        od = OD(video_name, options, exist_folder)
        anomaly_seg_num = od.outlier_detection()
        print('=-=-=-=-=-=-=-=-=-=-=-= Outlier Detection Done =-=-=-=-=-=-=-=-=-=-=-=')

    print('=-=-=-=-=-=-=-=-=-=-=-= Visualization Start =-=-=-=-=-=-=-=-=-=-=-=')
    viz = DR(options = options, basic_path=basic_path, video_name=video_name, od_result=anomaly_seg_num)
    fig = viz.run_DR()
    print('=-=-=-=-=-=-=-=-=-=-=-= Visualization Done =-=-=-=-=-=-=-=-=-=-=-=')
    return fig

def allowed_file(video_name):
    return '.' in video_name and \
        video_name.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

@app.callback(
    Output('selected_data', 'children'),
    Input('visualize_graph', 'selectedData'))
def display_selected_data(selectedData):
    global select_segment_list
    select_tmp = edit_json.get_seg_num(selectedData, "select")
    print("select tmp type:", type(select_tmp))
    print(select_tmp)
    if type(select_tmp) == list:
        select_segment_list = select_tmp
    return "Selected segment: {}".format(select_segment_list)

@app.callback(
    Output('btn_test_div', 'children'),
    Input('make_video_btn', 'n_clicks'))
def make_video_btn(n_clicks):
    global seg_btn_click
    if seg_btn_click < n_clicks:
        # Todo: feature_type, video_name
        seg_video_path = make_video.make_seg_video(select_segment_list, video_name, options['feature'])
        seg_btn_click = n_clicks
    #return 값 고치는 중
    return 'click button {} times!'.format(n_clicks)


application = DispatcherMiddleware(
    server,
    {"/plotly": app.server},
)

if __name__ == '__main__':
    run_simple('0.0.0.0', 5055, application, use_reloader=True)
    # server.run(
    #     host = '0.0.0.0',
    #     port = 5055,
    #     debug = True
    # )
    