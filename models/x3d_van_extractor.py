from models import X3D as x3d
from models import P3D as p3d
from models import C3D as c3d
from models import I3D as i3d
from models import slowfast as slowfast
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import convert_param,activity_name,pool_feature
import torchfile
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import os


'''X3D, P3D, C3D, I3D, SlowFast'''



#path = '/projects/vode/team1/I3D/kidnap/' #데이터 input path 지정
path = '/projects/vode/team1/BG_16frames/vandalism/'
target_dir = '/projects/vode/test_numpy_sec/blur_numpy_X3D/vandalism/' #numpy 저장 path 지정
net = x3d # 여기에 원하는 모델 쓰기 x3d, p3d, c3d, slowfast, i3d



if net == x3d :
    print("X3D is loading...")
    x3d_version = 'M'
    batch_size = 16
    zero_set = torch.zeros((1,3,16,224,224))
    trans = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])
    model = x3d.generate_model(x3d_version, base_bn_splits=1)
    model = model.cuda()
    weights=torch.load('./weightFile/x3d_charades_rgb_sgd_024000.pt')["model_state_dict"]
    model.load_state_dict(weights)
    #input_tensor = torch.autograd.Variable(torch.rand(1, 3, 16, 224, 224)).cuda() 
    #output = model(input_tensor)
    #print("X3D", output.shape)


elif net == c3d:
    print("C3D is loading...")
    batch_size = 16
    zero_set = torch.zeros((1,3,16,112,112))
    trans = transforms.Compose([transforms.Resize((112,112)),
                                transforms.ToTensor()])
    model = c3d.C3D(num_classes=400, pretrained=False)
    model = model.cuda()
    model.load_state_dict(convert_param(torchfile.load('./weightFile/c3d-sports1m-kinetics.t7'),
                                                    model.state_dict(),
                                                    verbose=False))
    #inputs = torch.rand(1, 3, 16, 112, 112).cuda()
    #outputs = model.forward(inputs)
    #print("C3D", outputs.size()) 


elif net == p3d:
    print("P3D is loading...")
    batch_size = 16
    zero_set = torch.zeros((1,3,16,160,160))
    trans = transforms.Compose([transforms.Resize((160,160)),
                                transforms.ToTensor()])
    model = p3d.P3D199(pretrained=True, num_classes=400)
    model = model.cuda()
    #data=torch.autograd.Variable(torch.rand(1,3,16,160,160)).cuda()   
    #out=model(data)
    #print("P3D",out.size())

elif net == slowfast:
    print("SlowFast is loading...")
    batch_size = 64
    zero_set = torch.zeros((1,3,64,224,224))
    trans = transforms.Compose([transforms.Resize((224,224)),   
                                transforms.ToTensor()])
    num_classes = 2
    model = slowfast.resnet50(class_num=num_classes)
    weights = torch.load('./weightFile/slowfast_weight.pth')['state_dict']

    for key in list(weights.keys()):
        if 'features.' in key:
            weights[key.replace('features.','')] = weights[key]
            del weights[key]
    model.load_state_dict(weights, strict=False)
    model = model.cuda()

elif net == i3d:
    print("I3D is loading...")
    batch_size = 64
    zero_set = torch.zeros((1,3,64,224,224))
    trans = transforms.Compose([transforms.Resize((224,224)),   
                                transforms.ToTensor()])
    num_classes = 2
    model = i3d.InceptionI3d(num_classes)   # only rgb
    model.replace_logits(157)
    weights = torch.load('./weightFile/rgb_charades.pt')
    model.load_state_dict(weights, strict=False)
    model = model.cuda()


if __name__ == "__main__":

    file_list = os.listdir(path)
    file_list.sort()#(reverse=True)

    for i in file_list:
        trainset = torchvision.datasets.ImageFolder(root = path + i,
					    transform = trans)
        print("data reading : "+path+i)

        trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=False, num_workers=2)

        make = zero_set

        for j,data in enumerate(trainloader):
            images,labels = data
            labels = labels.cpu() #labels : 한 영상 데이터 안에 있는 세그먼트 개수 
            labels = labels.detach().numpy() 

            #print(trainset.classes[labels[0]]) 
            #영상이름 + segment 정보   ex : 199-1_cam02_kidnap01_place01_day_summer_00010_normal
   


            images = images.transpose(1,0)

            if not images.shape[1] == batch_size:
                continue
            
            make[0][:][:][:][:] = images[:][:][:][:]


            
            out = model.forward(make.cuda())
            out = out.cpu()
            result = out.detach().numpy()

            #target_dir = '/home/proj_vode/data_numpy_C3D_re/kidnap/'+i+'/' #데이터 저장 path 지정
            save_dir = target_dir+i+'/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            np.save(save_dir+trainset.classes[labels[0]],result) 



