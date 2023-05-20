import os
from tkinter import image_names
os.environ['CUDA_VISIBLE_DEVICES']='0'
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import albumentations
import torch.optim as optim
from albumentations.pytorch import ToTensorV2, ToTensor
from ewslf.models.resnet import *
from ewslf.eval_model import *
from ewslf.compute_atten import *
from topk.svm import SmoothTop1SVM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)
torch.backends.cudnn.deterministic=True

mag = '10X'
dataset_name = 'sc'
model_path = './output'
img_dir = './Heatmap/'
atten_all = True
cut_width = 128
cut_length = 128
instance_loss_fn = SmoothTop1SVM(n_classes = 2)
save_path = model_path + f'/{mag}_{dataset_name}_checkpoint.pt'
if device.type == 'cuda':
    instance_loss_fn = instance_loss_fn.cuda()
model_ft = WSIClassifier(3, bn_track_running_stats=True, instance_loss_fn=instance_loss_fn)
model_ft = model_ft.to(device)

data_transforms = albumentations.Compose([
    ToTensor()
    ])    

optimizer = optim.Adam(model_ft.parameters(), lr=1e-4)

from ewslf.utils import *

model_ft, optimizer = load_ckp(save_path, model_ft, optimizer)

TEST_PATH = f'./{dataset_name}_dataset/{mag}_test_{dataset_name}.csv'
df_test = pd.read_csv(TEST_PATH)

_, pred_atten = eval_test(model_ft, df_test, data_transforms, compute_atten=True, atten_all=atten_all)
#compute attention map
atten_dict = compute_atten(pred_atten,dataset_name,mag,cut_width,cut_length)
# print(atten_dict)
from PIL import Image
for image_name in atten_dict.keys():
    plot_attention(atten_dict[image_name], image_name, img_dir=img_dir)
    