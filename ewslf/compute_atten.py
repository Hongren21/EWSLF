import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


def read_img(img_path, pred_atten, cut_width, cut_length, key):
    image = cv2.imread(img_path)
    img_h, img_w, img_c = image.shape
    num_width = int(img_h / cut_width)
    num_length = int(img_w / cut_length)
    image_atten = pred_atten[key].reshape(
        len(pred_atten[key]), num_width, num_length)
    return image_atten


def compute_atten(pred_atten, dataset_name, mag, cut_width, cut_length):
    atten_dict = {}
    for key in pred_atten:
        if dataset_name == 'sc':
            disease_list = ['squamous', 'bowen', 'basiloma']
            for disease_name in disease_list:
                disease_path = '..\\sc_dataset\\'+disease_name
                img_path = os.path.join(disease_path, str(key)+'-'+mag+'.tif')
                if os.path.exists(img_path):
                    image_atten = read_img(
                        img_path, pred_atten, cut_width, cut_length, key)
                    atten_dict[key] = image_atten
        elif dataset_name == 'rc':
            disease_list = ['KIRC', 'KIRP', 'KICH']
            for disease_name in disease_list:
                disease_path = '..\\rc_dataset\\' + 'TCGA-RC-' + \
                    str(mag) + '\\' + str(mag) + '\\' + disease_name
                img_path = os.path.join(disease_path, str(key)+'.png')
                if os.path.exists(img_path):
                    image_atten = read_img(
                        img_path, pred_atten, cut_width, cut_length, key)
                    atten_dict[key] = image_atten
    return atten_dict


def plot_attention(data, image_name, img_dir, X_label=None, Y_label=None):
    '''
        Plot the attention model heatmap
        Args:
        data: attn_matrix with shape [ty, tx], cutted before 'PAD'
        X_label: list of size tx, encoder tags
        Y_label: list of size ty, decoder tags
    '''
    for category in range(len(data)):
        att_group = data[category]

        fig, ax = plt.subplots(figsize=(20, 15))  # set figure size
        ax = sns.heatmap(att_group, cmap=plt.cm.Blues)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=40)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        fig.savefig(img_dir + f"{image_name}"+"_"+f"{category}"+".png")
        # Set axis labels
        if X_label != None and Y_label != None:
            X_label = [x_label for x_label in X_label]
            Y_label = [y_label for y_label in Y_label]

            xticks = range(0, len(X_label))
            ax.set_xticks(xticks, minor=False)  # major ticks
            # labels should be 'unicode'
            ax.set_xticklabels(X_label, minor=False, rotation=45)

            yticks = range(0, len(Y_label))
            ax.set_yticks(yticks, minor=False)
            # labels should be 'unicode'
            ax.set_yticklabels(Y_label, minor=False)

            ax.grid(True)
