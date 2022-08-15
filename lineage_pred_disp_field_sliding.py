avg = 0
for curr_index in np.unique(mask_pre)[1: ]:
    right = 0
    wrong = 0
    print("Index : ", curr_index)
    points = np.argwhere(mask_pre == curr_index)
    for point in points:
        z, y, x = point[0], point[1], point[2]
        disp_z = disp_z_map_pre[z][y][x]
        disp_y = disp_y_map_pre[z][y][x]
        disp_x = disp_x_map_pre[z][y][x]
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
