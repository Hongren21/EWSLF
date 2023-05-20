from ewslf.utils import *
import time
from topk.svm import SmoothTop1SVM
from ewslf.eval_model import *
from ewslf import train
from ewslf.models.resnet import *
from albumentations.pytorch import ToTensorV2, ToTensor
import torch.optim as optim
import albumentations
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)
torch.backends.cudnn.deterministic = True

mag = "10X"
dataset_name = "sc"
model_path = "./output"
if not os.path.exists(model_path):
    os.makedirs(model_path)
save_path = model_path + f"/{mag}_{dataset_name}_checkpoint.pt"
TRAIN_PATH = f"./{dataset_name}_dataset/{mag}_train_{dataset_name}.csv"
df = pd.read_csv(TRAIN_PATH)
df.head()

# Load Model
instance_loss_fn = SmoothTop1SVM(n_classes=2)
if device.type == "cuda":
    instance_loss_fn = instance_loss_fn.cuda()
model_ft = WSIClassifier(
    3, bn_track_running_stats=True, instance_loss_fn=instance_loss_fn
)
model_ft = model_ft.to(device)

data_transforms = albumentations.Compose([ToTensor()])
criterion_ce = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters(), lr=1e-4)
start = time.time()
# Train
model_ft = train.train_model(
    model_ft,
    criterion_ce,
    optimizer,
    df,
    data_transforms=data_transforms,
    fpath=save_path,
)
end1 = time.time()

model_ft, optimizer = load_ckp(save_path, model_ft, optimizer)
valid_images = dict(df.loc[df["is_valid"] == 1].groupby("wsi")[
                    "path"].apply(list))
valid_images_label = dict(
    df.loc[df["is_valid"] == 1].groupby("wsi")["label"].apply(max)
)
epoch_acc = eval_model(
    valid_images, valid_images_label, model_ft, data_transforms=data_transforms
)
end2 = time.time()
print("train time:{},test time:{}".format(end1 - start, end2 - end1))
