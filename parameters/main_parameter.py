# Directory
# 각자 바꾸기
# final_dir = '/home/yewon99/VizApp/code/viz_web/result/'
final_dir = '/projects/vode/VizApp/'
video_folder = final_dir + 'video/'
segment_folder = final_dir + 'segment/'
feature_folder = final_dir + 'feature/'
make_video_folder = final_dir + 'make_video/'


# Outlier Detection
basic_path = '/projects/vode/feature/I3D'
folder = '/kidnap'
outlier_percent = 0
outlier_seed = 777
test_split_seed = 777
n_cluster_start = 71 #default = 8 , if n_cluster_option == False, OV_n_cluster_option == False this is n_cluster
n_cluster_end = 60
use_weights = False
OV_n_cluster_option = False
folder_list = ['kidnap', 'assault', 'trespass']


# Dimension Reduction
n_neighbors = 9
min_dist = 0.01