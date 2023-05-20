import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import albumentations
from albumentations.pytorch import ToTensorV2, ToTensor

from ewslf.dataloader import *
from ewslf.eval_model import *
from ewslf.utils import *
from ewslf.cluster import run_clustering


def tensor_threshold(tensor, n, value):
    thresholds = [sorted(arr)[-n] for arr in tensor]
    return [[0 if x < th else value for x in arr]
            for arr, th in zip(tensor, thresholds)]


def train_model(model, criterion_ce, optimizer, df, data_transforms,
                num_cluster=8, num_img_per_cluster=16, num_epochs=50, fpath='checkpoint.pt', topk=False, bag_weight=0.7):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_images = dict(df.loc[df['is_valid'] == 0].groupby('wsi')[
                        'path'].apply(list))
    valid_images = dict(df.loc[df['is_valid'] == 1].groupby('wsi')[
                        'path'].apply(list))
    train_images_label = dict(
        df.loc[df['is_valid'] == 0].groupby('wsi')['label'].apply(max))
    valid_images_label = dict(
        df.loc[df['is_valid'] == 1].groupby('wsi')['label'].apply(max))

    if topk:
        num_cluster = 1
        num_img_per_cluster = 64

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_images, train_images_cluster, valid_images, valid_images_cluster = \
            run_clustering(train_images, valid_images, model, data_transforms=data_transforms,
                           num_cluster=num_cluster, topk=topk)

        if epoch > 0:
            print('NMI: {}'.format(
                cal_nmi(list(train_images_cluster.values()), train_images_cluster_last)))

        train_images_cluster_last = list(train_images_cluster.values()).copy()

        dataloaders, dataset_sizes = reinitialize_dataloader(train_images, train_images_cluster, train_images_label,
                                                             valid_images, valid_images_cluster, valid_images_label,
                                                             data_transforms=data_transforms, num_cluster=num_cluster,
                                                             num_img_per_cluster=num_img_per_cluster)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                epoch_acc = eval_model(
                    valid_images, valid_images_label, model, data_transforms=data_transforms)

                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    save_ckp(checkpoint, fpath)

                continue

            running_loss_wsi = 0.0
            running_corrects = 0
            optimizer.zero_grad()

            for i, (inputs, labels, inputs_cluster) in enumerate(dataloaders[phase]):

                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, Y_prob, Y_hat, _, instance_dict = model(
                        inputs, label=labels, instance_eval=True)  # TODO

                    _, preds = torch.max(outputs, 1)
                    loss_wsi = criterion_ce(outputs, labels)
                    instance_loss = instance_dict['instance_loss']

                    loss = bag_weight * loss_wsi + \
                        (1-bag_weight) * instance_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss_wsi += loss_wsi.item() * len(inputs)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss_wsi = running_loss_wsi / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss WSI: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss_wsi, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Final epoch model
    model_final = copy.deepcopy(model)

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model
