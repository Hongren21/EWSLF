import copy
import torch
import numpy as np
import pandas as pd
import albumentations
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2, ToTensor

from ewslf.dataloader import *
from ewslf.models.resnet import *


def eval_model(test_images, test_images_label, model, data_transforms):
    """ Pass all the patches in a WSI for validation prediction
    """
    pred_list = []
    pred_fname = []
    with torch.no_grad():
        for im, im_list in tqdm(test_images.items()):
            td = WSIDataloader(im_list, transform=data_transforms)
            tdl = torch.utils.data.DataLoader(td, batch_size=128,
                                              shuffle=False, num_workers=0)
            t_pred, _ = compute_attn_df(tdl, model)
            pred_list.append(t_pred)
            pred_fname.append(im)

    pred_df = pd.DataFrame({'wsi': pred_fname, 'prediction': pred_list})
    pred_df['actual'] = pred_df['wsi'].apply(lambda x: test_images_label[x])

    print('Test Accuracy: ', sum(
        pred_df['actual'] == pred_df['prediction'])/pred_df.shape[0])

    # Confusion Matrix
    label_map = {0: 'class1', 1: 'class2', 2: 'class3'}
    actual_label = pd.Series([label_map[x]
                             for x in pred_df['actual'].tolist()], name='Actual')
    predicted_label = pd.Series([label_map[x]
                                for x in pred_list], name='Predicted')
    print(pd.crosstab(actual_label, predicted_label))

    return sum(pred_df['actual'] == pred_df['prediction'])/pred_df.shape[0]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def compute_attn_df(tdl, model, track_running_stats=True):
    """ Pass through all the patches in a WSI for validation prediction
    """
    model = copy.deepcopy(model)
    enc_attn = EncAttn(model)
    enc_attn.eval()

    if track_running_stats:
        for m in enc_attn.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

    attn_rep = torch.tensor([])
    inputs_rep = torch.tensor([])
    fname_list = []

    for i, sample in enumerate(tdl):
        device = sample[0].device  # TODO
        attn_rep_instance, inp_rep_instance = enc_attn(sample[0].cuda())
        attn_rep_instance = attn_rep_instance.detach()
        inp_rep_instance = inp_rep_instance.detach()
        if len(inputs_rep) == 0:
            inputs_rep = inp_rep_instance
            attn_rep = attn_rep_instance
        else:
            inputs_rep = torch.cat((inputs_rep, inp_rep_instance))
            attn_rep = torch.cat((attn_rep, attn_rep_instance))
        fname_list += list(sample[1])

    A = torch.transpose(attn_rep, 1, 0)
    A = F.softmax(A, dim=1)
    M = torch.mm(A, inputs_rep)
    logits = torch.empty(1, model.n_classes).float().to(device)
    for c in range(model.n_classes):
        logits[0, c] = model.classifiers[c](M[c])  # TODO

    return torch.max(logits, 1)[1].item(), attn_rep.cpu()


def eval_test(model, df, data_transforms, compute_atten=False, atten_all=False):
    """ 
    Parameters:
        model - Trained model
        df - dataframe containing following columns:
            1. path - path of patches
            2. wsi - wsi identifier            
            optional - 
            3. label - positive or negative class - if label column 
            is not provided performance is not reported

    returns:
        df - dataframe with prediction for each WSI
    """

    test_images = dict(df.groupby('wsi')['path'].apply(list))
    pred_list = []
    pred_fname = []
    pred_attn = {}
    pred_dict = {}
    pred_path = []
    pred_fname1 = []
    pred_list1 = []
    with torch.no_grad():
        for im, im_list in tqdm(test_images.items()):
            if compute_atten:
                label = dict(df.groupby('wsi')['label'].apply(max))[im]
                import re
                im_list = sorted(im_list, key=lambda info: (
                    int(re.findall(r'_(\d+).', info)[0])))
            td = WSIDataloader(im_list, transform=data_transforms)
            tdl = torch.utils.data.DataLoader(td, batch_size=128,
                                              shuffle=False, num_workers=0)
            t_pred, attn_rep = compute_attn_df(tdl, model)
            if compute_atten:
                attn_rep = torch.transpose(attn_rep, 1, 0)
                attn_rep = torch.sigmoid(attn_rep)  # TODO
                if atten_all:
                    attn_rep = attn_rep.numpy()
                else:
                    attn_rep = torch.unsqueeze(attn_rep[label], dim=0).numpy()
                pred_dict[im] = t_pred
                pred_attn[im] = attn_rep
            pred_list += [t_pred]*len(im_list)
            pred_fname += [im]*len(im_list)
            pred_path += im_list
            pred_fname1.append(im)
            pred_list1.append(t_pred)

    pred_df = pd.DataFrame(
        {'wsi': pred_fname, 'prediction': pred_list, 'path': pred_path})

    if 'label' in df.columns:
        test_images_label = dict(df.groupby('wsi')['label'].apply(max))
        if compute_atten:
            print('true:', test_images_label)
            print('predict:', pred_dict)
        pred_df['actual'] = pred_df['wsi'].apply(
            lambda x: test_images_label[x])
        pred_wsi_df = pred_df[[
            'wsi', 'prediction', 'actual']].drop_duplicates()
        print('Test Accuracy: ', sum(
            pred_wsi_df['actual'] == pred_wsi_df['prediction'])/pred_wsi_df.shape[0])

    pred_df1 = pd.DataFrame({'wsi': pred_fname1, 'prediction': pred_list1})
    pred_df1['actual'] = pred_df1['wsi'].apply(lambda x: test_images_label[x])

    # Confusion Matrix
    label_map = {0: 'class1', 1: 'class2', 2: 'class3'}
    actual_label = pd.Series(
        [label_map[x] for x in pred_df1['actual'].tolist()], name='Actual')
    predicted_label = pd.Series([label_map[x]
                                for x in pred_list1], name='Predicted')
    print(pd.crosstab(actual_label, predicted_label))

    return pred_df, pred_attn
