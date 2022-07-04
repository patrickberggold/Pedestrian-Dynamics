import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def linear_interpolation(curr_point_or, lim_min_or, lim_max_or, lim_min_proj, lim_max_proj):
    return lim_min_proj + (curr_point_or - lim_min_or) * (lim_max_proj - lim_min_proj) / (lim_max_or - lim_min_or)

def read_txt_file(path_to_traj: str, delimiter: str = ','):
    assert os.path.isfile(path_to_traj)
    with open(path_to_traj, 'r') as f:
        # read lines and convert from str to float 
        lines = [[float(symbol) for symbol in line.strip().split('\t')] for line in f.readlines()]
    f.close()
    
    # sort by ped id, then timestamp
    lines.sort(key=lambda x: (x[1], x[0]))
    lines_np = np.array(lines)[:, 1:]

    pedIds = np.unique(lines_np[:,0])
    id_list = [[int(id) for id in np.argwhere(lines_np[:,0] == id_np)] for id_np in pedIds]       

    x_res, y_res = 800, 800
    img = np.ones((x_res,y_res,3))
    line_color = (1, 0, 0)

    xs = lines_np[:, 1]
    ys = lines_np[:, 2]
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    ped_Coords = {}
    for ped, pedIds in enumerate(id_list):
        ped_Coords.update({'ped_'+str(ped+1): []})
        for idPos in pedIds:
            x,y = lines_np[idPos][1:]
            x_proj = round(linear_interpolation(x, x_min, x_max, 0, x_res))
            y_proj = round(linear_interpolation(y, y_min, y_max, 0, y_res))
            ped_Coords['ped_'+str(ped+1)].append([x_proj, y_proj])

    for pedId in ped_Coords:
        if len(ped_Coords[pedId]) > 1:
            for coordId in range(1, len(ped_Coords[pedId])):
                start_point = (ped_Coords[pedId][coordId-1])
                end_point = (ped_Coords[pedId][coordId])

                cv2.line(img, start_point, end_point, color=line_color, thickness=1)
    plt.imshow(img)
    #plt.imshow(img.astype('uint8'), vmin=0,vmax=255)
    #cv2.imshow('Trajectories', img)


def read_trajs_from_seqArray(seq_array_np):
    pass
                

txt_filepath = os.path.join('\\'.join(['C:', 'Users', 'Remotey', 'Documents', 'Pedestrian-Dynamics', 'TrajectoryPrediction', 'Trajectory-Transformer', 'datasets', 'eth', 'train', 'crowds_zara02_train.txt']))
read_txt_file(txt_filepath, delimiter='\t')