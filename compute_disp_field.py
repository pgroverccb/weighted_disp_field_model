import json
import tifffile 
import numpy as np
import pickle

f = open("/mnt/ceph/users/pgrover/32_40_dataset/Lineages/GT_tracking_F32_to_F40.json", "r")
text = f.read()
text = json.loads(text)
text = text['G_based_on_nn']

def compute_disp_field(index):
    five_digit_str = str(index)
    while (len(five_digit_str) != 5):
        five_digit_str = '0' + five_digit_str
    input_pre = tifffile.imread(/mnt/ceph/users/pgrover/32_40_dataset/registered_images/image_reg_" + str(five_digit_str) + ".tif")
    input_pre = (input_pre - np.mean(input_pre))/np.std(input_pre)
    input_pre = input_pre - np.min(input_pre)
    input_pre = input_pre/np.max(input_pre)
    mask_pre = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(five_digit_str) + ".tif")

    five_digit_str = str(index + 1)
    while (len(five_digit_str) != 5):
        five_digit_str = '0' + five_digit_str
    input_post = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_images/image_reg_" + str(five_digit_str) + ".tif")
    input_post = (input_post - np.mean(input_post))/np.std(input_post)
    input_post = input_post - np.min(input_post)
    input_post = input_post/np.max(input_post)
    mask_post = tifffile.imread("/mnt/ceph/users/pgrover/32_40_dataset/registered_label_images/label_reg_" + str(five_digit_str) + ".tif")

    print("Index", index, "pre objects : ", np.unique(mask_pre))
    print("post objects : ", np.unique(mask_post))

    mapping_pre = {}
    for i in range(len(text['Edges'])):
        if (text['Edges'][i]['EndNodes'][0][:3] == str(index)):
            if (int(text['Edges'][i]['EndNodes'][0][4:7]) not in mapping_pre.keys()):
                mapping_pre[int(text['Edges'][i]['EndNodes'][0][4:7])] = []
            # print(text['Edges'][i]['EndNodes'][0], text['Edges'][i]['EndNodes'][1])
            mapping_pre[int(text['Edges'][i]['EndNodes'][0][4:7])].append(int(text['Edges'][i]['EndNodes'][1][4:7]))


    disp_x_map_pre = np.zeros((160, 392, 240))
    disp_y_map_pre = np.zeros((160, 392, 240))
    disp_z_map_pre = np.zeros((160, 392, 240))

    dividing_list = []
    constant_list = []

    for i in mapping_pre.keys():
        if (len(mapping_pre[i]) == 2):
            dividing_list.append(i)
        else:
            constant_list.append(i)

    for dividing_index in dividing_list:
        print("Dividing Index : ", dividing_index)
        all_points = np.argwhere(mask_pre == dividing_index)
        count_parent = len(all_points)
        z_parent_cent, y_parent_cent, x_parent_cent = np.mean(np.argwhere(mask_pre == dividing_index), axis = 0)
        print("Parent Centroid : ", round(z_parent_cent, 3), round(y_parent_cent, 3), round(x_parent_cent, 3))

        first = mapping_pre[dividing_index][0]
        second = mapping_pre[dividing_index][1]

        all_points = np.argwhere(mask_post == first)
        count_daughter_0 = len(all_points)
        z_daughter_0_cent, y_daughter_0_cent, x_daughter_0_cent = np.mean(np.argwhere(mask_post == first), axis = 0)
        print("Daughter 0 Centroid : ", round(z_daughter_0_cent, 3), round(y_daughter_0_cent, 3), round(x_daughter_0_cent, 3))

        all_points = np.argwhere(mask_post == second)
        count_daughter_1 = len(all_points)
        z_daughter_1_cent, y_daughter_1_cent, x_daughter_1_cent = np.mean(np.argwhere(mask_post == second), axis = 0)
        print("Daughter 1 Centroid : ", round(z_daughter_1_cent, 3), round(y_daughter_1_cent, 3), round(x_daughter_1_cent, 3))

        portion_to_daughter_0 = 0
        portion_to_daughter_1 = 0

        # z_parent_bbox_min, y_parent_bbox_min, x_parent_bbox_min = np.min(np.argwhere(custom_label_map[0] == dividing_index), axis = 0)
        # z_parent_bbox_max, y_parent_bbox_max, x_parent_bbox_max = np.max(np.argwhere(custom_label_map[0] == dividing_index), axis = 0)

        bisector_z = (z_daughter_0_cent + z_daughter_1_cent) / 2
        bisector_y = (y_daughter_0_cent + y_daughter_1_cent) / 2
        bisector_x = (x_daughter_0_cent + x_daughter_1_cent) / 2

        normal_z = z_parent_cent - bisector_z
        normal_y = y_parent_cent - bisector_y
        normal_x = x_parent_cent - bisector_x

        dirs = []
        for point in all_points:
            dir = (normal_z * point[0]) + (normal_y * point[1]) + (normal_x * point[2])
            dirs.append(dir)
        dirs = np.array(dirs)
        plane_thresh = np.mean(dirs)
        x = 0
        print("Plane Constant : ", plane_thresh)
        for i in dirs:
            if (i > plane_thresh):
                x += 1
        print("Portion : ", x/len(dirs))
        
        all_points = np.argwhere(mask_pre == dividing_index)
        
        for point in all_points:
            # dir = (normal_z * (point[0] - z_parent_cent)) + (normal_y * (point[1] - y_parent_cent)) + (normal_x * (point[2] - x_parent_cent))
            dir = (normal_z * point[0]) + (normal_y * point[1]) + (normal_x * point[2])
            
            # dir = (point[2] - bisector_x)*1.0/l_x + (point[1] - bisector_y)*1.0/m_y + (point[0] - bisector_z)*1.0/n_z
            # dirs.append(dir)
            if (dir > plane_thresh):
                portion_to_daughter_0 += 1
                # dist_to_daughter_0 = pow(pow(z_daughter_0_cent * 10.0 - point[0] * 10.0, 2) + pow(y_daughter_0_cent - point[1], 2) + pow(x_daughter_0_cent - point[2], 2), 0.5)
                disp_z_map_pre[point[0], point[1], point[2]] += z_daughter_0_cent - point[0]
                disp_y_map_pre[point[0], point[1], point[2]] += y_daughter_0_cent - point[1]
                disp_x_map_pre[point[0], point[1], point[2]] += x_daughter_0_cent - point[2]
      
            else:
                portion_to_daughter_1 += 1
                # dist_to_daughter_1 = pow(pow(z_daughter_1_cent * 10.0 - point[0] * 10.0, 2) + pow(y_daughter_1_cent - point[1], 2) + pow(x_daughter_1_cent - point[2], 2), 0.5)
                disp_z_map_pre[point[0], point[1], point[2]] += z_daughter_1_cent - point[0]
                disp_y_map_pre[point[0], point[1], point[2]] += y_daughter_1_cent - point[1]
                disp_x_map_pre[point[0], point[1], point[2]] += x_daughter_1_cent - point[2]
      
    for constant_index in constant_list[:-1]:
        print("Constant Index : ", constant_index)
        
        all_points = np.argwhere(mask_pre == constant_index)
        z_pre_cent, y_pre_cent, x_pre_cent = np.mean(np.argwhere(mask_pre == constant_index), axis = 0)
        print("Pre Movement Centroid : ", round(z_pre_cent, 3), round(y_pre_cent, 3), round(x_pre_cent, 3))

        all_post_points = np.argwhere(mask_post == mapping_pre[constant_index])
        z_post_cent, y_post_cent, x_post_cent = np.mean(np.argwhere(mask_post == mapping_pre[constant_index][0]), axis = 0)
        print("Post Movement Centroid : ", round(z_post_cent, 3), round(y_post_cent, 3), round(x_post_cent, 3))
        
        all_points = np.argwhere(mask_pre == constant_index)
        z_pre_cent, y_pre_cent, x_pre_cent = np.mean(np.argwhere(mask_pre == constant_index), axis = 0)
        # print("Pre Movement Centroid : ", round(z_pre_cent, 3), round(y_pre_cent, 3), round(x_pre_cent, 3))

        for point in all_points:
            disp_z_map_pre[point[0], point[1], point[2]] += z_post_cent - point[0]
            disp_y_map_pre[point[0], point[1], point[2]] += y_post_cent - point[1]
            disp_x_map_pre[point[0], point[1], point[2]] += x_post_cent - point[2]

    return input_pre, input_post, disp_x_map_pre, disp_y_map_pre, disp_z_map_pre
