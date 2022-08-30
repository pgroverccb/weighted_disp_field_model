import numpy as np
import torch
from tqdm import tqdm
from pytorch_connectomics.connectomics.model.arch.unet import UNet3D
from monai.inferers import sliding_window_inference
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage
from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import dilation
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    ToTensord,
)
from monai.data import (
    DataLoader,
    CacheDataset,
)
import os
import pickle
import tifffile

dataset_path = "sample_dataset/"
test_set_paths_pairs = [['image_reg_00135.tif', 'image_reg_00136.tif']]
saved_weights_path = "utils/u3d_bcd_saved_model.pth"
patch_size = (16, 128, 128)
                        
test_files = []

for pair in test_set_paths_pairs:
      input_volume_1 = tifffile.imread(dataset_path + pair[0])
      input_volume_2 = tifffile.imread(dataset_path + pair[1])

      orig_shape = input_volume_1.shape
      input_volume_1 = (input_volume_1 - np.mean(input_volume_1))/(np.std(input_volume_1))
      scaler = MinMaxScaler()
      scaler.fit(input_volume_1.flatten().reshape(-1, 1))
      input_volume_1 = scaler.transform(input_volume_1.flatten().reshape(-1, 1)).reshape(orig_shape)

      orig_shape = input_volume_2.shape
      input_volume_2 = (input_volume_2 - np.mean(input_volume_2))/(np.std(input_volume_2))
      scaler = MinMaxScaler()
      scaler.fit(input_volume_2.flatten().reshape(-1, 1))
      input_volume_2 = scaler.transform(input_volume_2.flatten().reshape(-1, 1)).reshape(orig_shape)

      image_path = dataset_path + "normalized_image.npy"
      test_files.append({'image' : image_path})
      np.save(image_path, np.array([input_volume_1, input_volume_2]))
      print("Completed operation for ", pair)

test_files = np.array(test_files)
test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
)

test_ds = CacheDataset(
    data=test_files, 
    transform=test_transforms,
    cache_num=1, cache_rate=0.0, num_workers=2
)

test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
)

test_iterator = tqdm(test_loader, desc="Testing (X / X Steps) (dice=X.X)", dynamic_ncols=True)
model = UNet3D()
model.load_state_dict(torch.load(saved_weights_path))
model = model.cuda()
model.eval()

def getSegType(mid):
    m_type = np.uint64
    return m_type

def cast2dtype(segm):
    max_id = np.amax(np.unique(segm))
    m_type = getSegType(int(max_id))
    return segm.astype(m_type)
    
def remove_small_instances(segm: np.ndarray, 
                           thres_small: int = 128, 
                           mode: str = 'background'):
    assert mode in ['none', 
                    'background', 
                    'background_2d', 
                    'neighbor',
                    'neighbor_2d']
    if mode == 'none':
        return segm
    if mode == 'background':
        return remove_small_objects(segm, thres_small)
    elif mode == 'background_2d':
        temp = [remove_small_objects(segm[i], thres_small)
                for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)

    if mode == 'neighbor':
        return merge_small_objects(segm, thres_small, do_3d=True)
    elif mode == 'neighbor_2d':
        temp = [merge_small_objects(segm[i], thres_small)
                for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)

def merge_small_objects(segm, thres_small, do_3d=False):
    struct = np.ones((1,3,3)) if do_3d else np.ones((3,3))
    indices, counts = np.unique(segm, return_counts=True)
    for i in range(len(indices)):
        idx = indices[i]
        if counts[i] < thres_small:
            temp = (segm == idx).astype(np.uint8)
            coord = bbox_ND(temp, relax=2)
            cropped = crop_ND(temp, coord)

            diff = dilation(cropped, struct) - cropped
            diff_segm = crop_ND(segm, coord)
            diff_segm[np.where(diff==0)]=0

            u, ct = np.unique(diff_segm, return_counts=True)
            if len(u) > 1 and u[0] == 0:
                u, ct = u[1:], ct[1:]
            segm[np.where(segm==idx)] = u[np.argmax(ct)]
    return segm

def bcd_watershed(semantic, boundary, distance, thres1=0.9, thres2=0.8, thres3=0.85, thres4=0.5, thres5=0.0, thres_small=128, 
                  scale_factors=(1.0, 1.0, 1.0), remove_small_mode='background', seed_thres=32, return_seed=False):
    # seed_map = (semantic > thres1) * (boundary < thres2) * (distance > thres4)
    # foreground = (semantic > thres3) * (distance > thres5)
    seed_map = (semantic > thres1)
    foreground = (semantic > thres3)
    seed = label(seed_map)
    seed = remove_small_objects(seed, seed_thres)
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
        
    if not return_seed:
        return cast2dtype(segm)

    return cast2dtype(segm), seed

with torch.no_grad():
    for step, batch in enumerate(test_iterator):
        val_inputs = batch["image"].cuda()
        # print("Shape of val_inputs : ", val_inputs.shape)
        val_outputs = []
        
        val_outputs_1 = sliding_window_inference(val_inputs[:, :, 0], patch_size, 4, model)
        val_outputs_2 = sliding_window_inference(val_inputs[:, :, 1], patch_size, 4, model)

        # print("Shape of val_outputs_1 : ", val_outputs_1.shape)
        # print("Shape of val_outputs_2 : ", val_outputs_2.shape)

        val_outputs = np.array([val_outputs_1[0].detach().cpu().numpy(), val_outputs_2[0].detach().cpu().numpy()])

        np.save("utils/predicted_u3d_map.npy", val_outputs)

        out_1 = bcd_watershed(val_outputs_1[0, 0].detach().cpu().numpy(), val_outputs_1[0, 1].detach().cpu().numpy(), val_outputs_1[0, 2].detach().cpu().numpy(), thres1 = 15, thres2 = -40, thres3 = 5, thres4 = -15, thres5 = -0.3, thres_small = 64, seed_thres = 16)
        out_2 = bcd_watershed(val_outputs_2[0, 0].detach().cpu().numpy(), val_outputs_2[0, 1].detach().cpu().numpy(), val_outputs_2[0, 2].detach().cpu().numpy(), thres1 = 15, thres2 = -40, thres3 = 5, thres4 = -15, thres5 = -0.3, thres_small = 64, seed_thres = 16)
        out = np.array([out_1, out_2])

        np.save("utils/predicted_seg_mask.npy", out)
        print("Found", np.unique(out_1), "unique objects in pre frame.")
        print("Found", np.unique(out_2), "unique objects in post frame.")
