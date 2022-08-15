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

class Dataset(torch.utils.data.Dataset):
  def __init__(self, list_IDs):
        self.list_IDs = list_IDs

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        ID = self.list_IDs[index]
        print("Loading ID : ", ID)
        file = open("/mnt/ceph/users/pgrover/disp_field_dataset/sample_" + str(ID) + ".pkl", 'rb')
        sample_full = pickle.load(file)
        input = sample_full['input']
        output = sample_full['output']
        output[2:] = output[2:]/20.0
        z_offset = random.randint(5, 20)
        y_offset = random.randint(80, 120)
        x_offset = random.randint(45, 85)
        input = input[:, z_offset : z_offset + 128, y_offset : y_offset + 128, x_offset : x_offset + 128]
        output = output[:, z_offset : z_offset + 128, y_offset : y_offset + 128, x_offset : x_offset + 128]
        return input, output

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 0}
max_epochs = 10

# Datasets
partition = {'train' : [], 'validation' : []}
for i in range(50, 130):
    prob = random.random()
    if (prob > 0.85):
        partition['validation'].append(i)
    else:
        partition['train'].append(i)

# Generators
training_set = Dataset(partition['train'])
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'])
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

disp_field_model = UNet3D(in_channel = 2, out_channel = 5, is_isotropic = True)
disp_field_model = disp_field_model.cuda()

mse_loss = nn.L1Loss()
optimizer = torch.optim.Adam(disp_field_model.parameters(), lr=0.0001)
print("Begin training.")
for e in range(1, 1000+1):
    disp_field_model.train()
    batch_num = 0
    train_loss_avg = 0.0
    val_loss_avg = 0.0

    for X_train_batch, y_train_batch in training_generator:
        batch_num += 1
        X_train_batch, y_train_batch = X_train_batch.to('cuda', dtype = torch.float), y_train_batch.to('cuda', dtype = torch.float)
        optimizer.zero_grad()
        y_train_pred = disp_field_model(X_train_batch)
        
        div_binary_mask = (y_train_batch[0, 0, :, :, :].clone() > 0.9)
        non_binary_mask = (y_train_batch[0, 0, :, :, :].clone() < 0.9)

        l2_x_non = mse_loss(y_train_pred[0, 2, :, :, :] * non_binary_mask, y_train_batch[0, 2, :, :, :] * non_binary_mask)
        l2_y_non = mse_loss(y_train_pred[0, 3, :, :, :] * non_binary_mask, y_train_batch[0, 3, :, :, :] * non_binary_mask)
        l2_z_non = mse_loss(y_train_pred[0, 4, :, :, :] * non_binary_mask, y_train_batch[0, 4, :, :, :] * non_binary_mask)
        
        l2_x_div = mse_loss(y_train_pred[0, 2, :, :, :] * div_binary_mask, y_train_batch[0, 2, :, :, :] * div_binary_mask)
        l2_y_div = mse_loss(y_train_pred[0, 3, :, :, :] * div_binary_mask, y_train_batch[0, 3, :, :, :] * div_binary_mask)
        l2_z_div = mse_loss(y_train_pred[0, 4, :, :, :] * div_binary_mask, y_train_batch[0, 4, :, :, :] * div_binary_mask)
        
        print("L2 x div : ", round(l2_x_div.item(), 3), "| L2 y div : ", round(l2_y_div.item(), 3), "| L2 z div : ", round(l2_z_div.item(), 3))
        print("L2 x non div : ", round(l2_x_non.item(), 3), "| L2 y non div: ", round(l2_y_non.item(), 3), "| L2 z non div : ", round(l2_z_non.item(), 3))

        train_loss = l2_x_div + l2_y_div + l2_z_div + 10 * (l2_x_non + l2_y_non + l2_z_non)
        train_loss_avg += train_loss.item()
        train_loss.backward()
        optimizer.step()

    for X_val_batch, y_val_batch in validation_generator:
        batch_num += 1
        X_val_batch, y_val_batch = X_val_batch.to('cuda', dtype = torch.float), y_val_batch.to('cuda', dtype = torch.float)
        optimizer.zero_grad()
        y_val_pred = disp_field_model(X_val_batch)
        
        div_binary_mask = (y_val_batch[0, 0, :, :, :].clone() > 0.9)
        non_binary_mask = (y_val_batch[0, 0, :, :, :].clone() < 0.9)
        
        l2_x_non = mse_loss(y_val_pred[0, 2, :, :, :] * non_binary_mask, y_val_batch[0, 2, :, :, :] * non_binary_mask)
        l2_y_non = mse_loss(y_val_pred[0, 3, :, :, :] * non_binary_mask, y_val_batch[0, 3, :, :, :] * non_binary_mask)
        l2_z_non = mse_loss(y_val_pred[0, 4, :, :, :] * non_binary_mask, y_val_batch[0, 4, :, :, :] * non_binary_mask)
        
        l2_x_div = mse_loss(y_val_pred[0, 2, :, :, :] * div_binary_mask, y_val_batch[0, 2, :, :, :] * div_binary_mask)
        l2_y_div = mse_loss(y_val_pred[0, 3, :, :, :] * div_binary_mask, y_val_batch[0, 3, :, :, :] * div_binary_mask)
        l2_z_div = mse_loss(y_val_pred[0, 4, :, :, :] * div_binary_mask, y_val_batch[0, 4, :, :, :] * div_binary_mask)
        
        print("L2 x div : ", round(l2_x_div.item(), 3), "| L2 y div : ", round(l2_y_div.item(), 3), "| L2 z div : ", round(l2_z_div.item(), 3))
        print("L2 x non div : ", round(l2_x_non.item(), 3), "| L2 y non div: ", round(l2_y_non.item(), 3), "| L2 z non div : ", round(l2_z_non.item(), 3))

        val_loss = l2_x_div + l2_y_div + l2_z_div + 10 * (l2_x_non + l2_y_non + l2_z_non)
        val_loss_avg += val_loss.item()
        optimizer.step()
    print("Epoch : ", e, "Train Loss : ", round(train_loss_avg/len(partition['train']), 3), "Val Loss : ", round(val_loss_avg/len(partition['validation']), 3))
