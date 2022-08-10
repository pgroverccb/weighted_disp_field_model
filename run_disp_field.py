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

class Dataset(torch.utils.data.Dataset):
  def __init__(self, list_IDs):
        self.list_IDs = list_IDs

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        ID = self.list_IDs[index]
        # print("Loading ID : ", ID)
        input = np.load("/mnt/ceph/users/pgrover/disp_field_dataset/inputs/" + str(ID + 1) + ".npy")
        input = np.zeros((2, 128, 128, 128))
        output = np.load("/mnt/ceph/users/pgrover/disp_field_dataset/outputs/" + str(ID + 1) + ".npy")
        output = output.reshape((5, 128, 128, 128))
        # output = output/100.0
        return input, output

# Parameters
params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 10

# Datasets
partition = {'train' : [], 'validation' : []}
for i in range(0, 140):
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
        # optimizer.zero_grad()
        y_train_pred = disp_field_model(X_train_batch)
        l2_x = mse_loss(y_train_pred[0, 0, :, :, :], y_train_batch[0, 0, :, :, :])
        l2_y = mse_loss(y_train_pred[0, 1, :, :, :], y_train_batch[0, 1, :, :, :])
        l2_z = mse_loss(y_train_pred[0, 2, :, :, :], y_train_batch[0, 2, :, :, :])
        # print("L1 x : ", round(l2_x.item(), 3), "| L1 y: ", round(l2_y.item(), 3), "| L1 z: ", round(l2_z.item(), 3))
        train_loss = l2_x + l2_y + l2_z
        train_loss_avg += train_loss.item()
        # train_loss.backward()
        # optimizer.step()

    for X_val_batch, y_val_batch in validation_generator:
        batch_num += 1
        X_val_batch, y_val_batch = X_val_batch.to('cuda', dtype = torch.float), y_val_batch.to('cuda', dtype = torch.float)
        # optimizer.zero_grad()
        y_val_pred = disp_field_model(X_val_batch)
        l2_x = mse_loss(y_val_pred[0, 0, :, :, :], y_val_batch[0, 0, :, :, :])
        l2_y = mse_loss(y_val_pred[0, 1, :, :, :], y_val_batch[0, 1, :, :, :])
        l2_z = mse_loss(y_val_pred[0, 2, :, :, :], y_val_batch[0, 2, :, :, :])
        # print("L1 x : ", round(l2_x.item(), 3), "| L1 y: ", round(l2_y.item(), 3), "| L1 z: ", round(l2_z.item(), 3))
        val_loss = l2_x + l2_y + l2_z
        val_loss_avg += val_loss.item()
        # optimizer.step()
    print("Epoch : ", e, "Train Loss : ", round(train_loss_avg/len(partition['train']), 3), "Val Loss : ", round(val_loss_avg/len(partition['validation']), 3))
