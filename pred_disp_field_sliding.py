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

# Datasets
testing_partition = []
for i in range(130, 133):
    testing_partition.append(i)

disp_field_model = UNet3D(in_channel = 2, out_channel = 5, is_isotropic = True)
disp_field_model = disp_field_model.cuda()
disp_field_model.load_state_dict(torch.load("/mnt/home/pgrover/weighted_disp_field_model/utils/saved_model.pth"))
 
print("Begin Testing.")
for ID in testing_partition:
    file = open("/mnt/ceph/users/pgrover/growth_field_dataset/sample_" + str(ID) + ".pkl", 'rb')
    sample_full = pickle.load(file)
    input = sample_full['input']
    output = sample_full['output']
    input = input[:, 13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
    output = input[:, 13 : 13 + 128, 64 : 64 + 128, 96 : 96 + 128]
    input, output = input.to('cuda', dtype = torch.float), output.to('cuda', dtype = torch.float)
    output_pred = disp_field_model(input)
    np.save("/mnt/home/pgrover/weighted_disp_field_model/utils/testing_pred_" + str(ID) + ".npy", output_pred.detach().cpu().numpy())
