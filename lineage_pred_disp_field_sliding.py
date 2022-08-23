from pytorch_connectomics.connectomics.model.arch.unet import UNet3D
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import tifffile
import json
import random
import pickle

f = open("/mnt/ceph/users/pgrover/32_40_dataset/Lineages/GT_tracking_F32_to_F40.json", "r")
text = f.read()
text = json.loads(text)
text = text['G_based_on_nn']

mapping_pre = {}
    for i in range(len(text['Edges'])):
        if (text['Edges'][i]['EndNodes'][0][:3] == str(index)):
            if (int(text['Edges'][i]['EndNodes'][0][4:7]) not in mapping_pre.keys()):
                mapping_pre[int(text['Edges'][i]['EndNodes'][0][4:7])] = []
            # print(text['Edges'][i]['EndNodes'][0], text['Edges'][i]['EndNodes'][1])
            mapping_pre[int(text['Edges'][i]['EndNodes'][0][4:7])].append(int(text['Edges'][i]['EndNodes'][1][4:7]))
            
testing_partition = []
for i in range(130, 133):
    testing_partition.append(i)
    
for index in testing_partition:
    five_digit_str = str(index)
    while (len(five_digit_str) != 5):
        five_digit_str = '0' + five_digit_str
    mask_pre = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(five_digit_str) + ".tif")

    five_digit_str = str(index + 1)
    while (len(five_digit_str) != 5):
        five_digit_str = '0' + five_digit_str
    mask_post = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(five_digit_str) + ".tif")

    disp_field_pred = np.load("/mnt/home/pgrover/weighted_disp_field_model/utils/testing_pred_" + str(index) + ".npy")
    disp_z_map, disp_y_map, disp_x_map = disp_field_pred[2], disp_field_pred[3], disp_field_pred[4]
    
    avg = 0
    for curr_index in np.unique(mask_pre)[1: ]:
        right = 0
        wrong = 0
        print("Index : ", curr_index)
        points = np.argwhere(mask_pre == curr_index)
        for point in points:
            z, y, x = point[0], point[1], point[2]
            disp_z = disp_z_map[z][y][x]
            disp_y = disp_y_map[z][y][x]
            disp_x = disp_x_map[z][y][x]
            curr_nuc = mask_pre[z][y][x]
            z_new = round(disp_z + z)
            y_new = round(disp_y + y)
            x_new = round(disp_x + x)    
            if (len(mapping_pre[mask_pre[z][y][x]]) == 2):
                if (mapping_pre[mask_pre[z][y][x]][0] == mask_post[z_new][y_new][x_new] or mapping_pre[mask_pre[z][y][x]][1] == mask_post[z_new][y_new][x_new]):
                    right += 1
                else:
                    wrong += 1
            else:
                if (mapping_pre[mask_pre[z][y][x]][0] == mask_post[z_new][y_new][x_new]):
                    right += 1
                else:
                    wrong += 1
        avg += round(right/(right + wrong) * 100.0, 2)
        print("Current Accuracy : ", round(right/(right + wrong) * 100.0, 2))
    print("Average Accuracy : ", round(avg/len(np.unique(mask_pre)[1: ]), 3))
