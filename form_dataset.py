import json
import tifffile 
import numpy as np
import pickle
from compute_disp_field import compute_disp_field

f = open("/content/drive/MyDrive/GT_tracking_F32_to_F40.json", "r")
text = f.read()
text = json.loads(text)
text = text['G_based_on_nn']

for index in range(50, 133):
    input_pre, input_post, disp_x_map_pre, disp_y_map_pre, disp_z_map_pre = compute_disp_field(index)
    file = open("/content/drive/MyDrive/supporting_files_cont_cell_cycle_pred/growth_dataset_sample/sample_" + str(index) + ".pkl", 'rb')
    sample = pickle.load(file)
    growth_output = sample['output']
        
    data_inputs = np.array([input_pre, input_post])
    data_outputs = np.array([growth_output[0], growth_output[1], disp_x_map_pre, disp_y_map_pre, disp_z_map_pre])
    sample_full = {'input' : data_inputs, 'output' : data_outputs}
    file_pointer = open("/content/drive/MyDrive/supporting_files_disp_field_model/disp_field_dataset/sample_" + str(index) + ".pkl", "wb")
    pickle.dump(sample_full, file_pointer)
