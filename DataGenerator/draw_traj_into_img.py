import os
import gzip
import shutil
import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import h5py
from skimage.draw import line

# Input: current point coordinate, min floorplan coord, max floorplan coord, min resolution coord, max resolution coord -> output: projected coordinated
def linear_interpolation(curr_point_or, lim_min_or, lim_max_or, lim_min_proj, lim_max_proj):
    return lim_min_proj + (curr_point_or - lim_min_or) * (lim_max_proj - lim_min_proj) / (lim_max_or - lim_min_or)

def get_color_from_colormap(index, max_index):
    # in style of https://jakevdp.github.io/PythonDataScienceHandbook/04.07-customizing-colorbars.html
    cmap = plt.cm.get_cmap('jet')
    start, end = 65, 200
    # range = np.arange(cmap.N)
    colors = cmap(np.arange(start, end))
    assert 0 <= index / max_index <= 1
    colors = colors[int(index / max_index * (end - start))][:-1]*255
    # colors = colors[int(index / max_index * cmap.N)][:-1]*255
    return colors

def get_color_from_array(index, max_index, return_in_cv2: bool = False):

    # colors = [get_color_from_array(i, 179) for i in range(179)]
    # [143, 225, 255] -> [0, 187, 255] -> [0, 0, 255] -> [180, 0, 255]
    r_val_1, g_val_1 = 143, 225
    r_val_2, g_val_2 = 0, 187
    r_val_3, g_val_3 = 0, 0
    r_val_4, g_val_4 = 180, 0
    b_val = 255 
    color_range_1_rval = np.arange(143, 0, -1)
    color_range_2_gval = np.arange(187, 0, -1)
    color_range_3_rval = np.arange(181)

    color_range_len = len(color_range_1_rval) + len(color_range_2_gval) + len(color_range_3_rval)

    fraction1 = len(color_range_1_rval)/color_range_len # 0.27984344422700586
    fraction2 = (len(color_range_1_rval) + len(color_range_2_gval))/color_range_len # 0.6457925636007827

    fr = index / max_index
    if fr <= fraction1:
        r_val = round(r_val_1 - index / (fraction1*max_index) * (r_val_1 - r_val_2))
        g_val = round(g_val_1 - index / (fraction1*max_index) * (g_val_1 - g_val_2))
    elif fraction1 < fr <= fraction2:
        ll = (index-fraction1*max_index) / (fraction2*max_index-fraction1*max_index)
        r_val = 0
        g_val = round(g_val_2 - (index-fraction1*max_index) / (fraction2*max_index-fraction1*max_index) * (g_val_2 - g_val_3))
    elif fraction2 <= fr <= 1:
        ll = (index-fraction2*max_index) / (max_index-fraction2*max_index)
        g_val = 0
        r_val = round(r_val_3 + (index-fraction2*max_index) / (max_index-fraction2*max_index) * (r_val_4 - r_val_3))
    else:
        raise ValueError

    color_array = np.array([r_val, g_val, b_val])

    return color_array.astype('float64')

def get_color_from_pedId(id, num_agents = 40):
    id = int(id)
    if id < 5:
        color = np.array([138, 255, 0])
    elif 5 <= id < 10:
        color = np.array([0, 255, 171])
    elif 10 <= id < 15:
        color = np.array([0, 255, 255])
    elif 15 <= id < 20:
        color = np.array([0, 160, 255])
    elif 20 <= id < 25:
        color = np.array([0, 57, 255])
    elif 25 <= id < 30:
        color = np.array([109, 0, 255])
    elif 30 <= id < 35:
        color = np.array([185, 0, 255])
    elif 35 <= id <= 40:
        color = np.array([255, 0, 213])
    else:
        raise NotImplementedError

    return color.astype('float64')

def get_customized_colormap():

    start = 65
    end = 200
    jet = cm.get_cmap('jet', end-start)
    custom_colors = jet(np.linspace(0, 1, end-start))*255
    # define colors
    # for idx in range(start, end):
    #     custom_colors[idx] = class_names[key]
    customed_colormap = ListedColormap(custom_colors)
    return customed_colormap

min_x_dxf = -0.15
min_y_dxf = -0.15
max_x_dxf = 20.15
max_y_dxf = 20.15

base_path = "C:\\Users\\ga78jem\\Documents\\"

floorplans_path = os.path.join(base_path, "Revit\\Exports")
crowdit_simulations_path = os.path.join(base_path, "Crowdit")

maximum_overall_time = 0 # for the 27 simulations = 89.5 seconds or 179 half seconds
MAX_TIME = 89.5 # seconds
MAX_TIMESTAMPS = int(MAX_TIME*2)

HDF5_ROOT_FOLDER = 'C:\\Users\\ga78jem\\Documents\\Revit\\HDF5_INPUT_IMAGES_resolution_800_800'
REVIT_ROOT_FOLDER = 'C:\\Users\\ga78jem\\Documents\\Revit\\Exports'

HDF5_COLOR_TRAJ_PATH = 'C:\\Users\\ga78jem\\Documents\\Revit\\HDF5_GT_COLORED_TRAJ_resolution_800_800'
if not os.path.isdir(HDF5_COLOR_TRAJ_PATH): os.mkdir(HDF5_COLOR_TRAJ_PATH)

HDF5_TRAJ_MASK_PATH = 'C:\\Users\\ga78jem\\Documents\\Revit\\HDF5_GT_TIMESTAMP_MASKS_resolution_800_800'
if not os.path.isdir(HDF5_TRAJ_MASK_PATH): os.mkdir(HDF5_TRAJ_MASK_PATH)

HDF5_TIME_AND_ID_MASK_PATH = 'C:\\Users\\ga78jem\\Documents\\Revit\\HDF5_GT_TIME_AND_ID_MASKS_resolution_800_800'
if not os.path.isdir(HDF5_TIME_AND_ID_MASK_PATH): os.mkdir(HDF5_TIME_AND_ID_MASK_PATH)

for floorplan_folder in os.listdir(HDF5_ROOT_FOLDER):
    h5_img_traj_floorplan_folder = os.path.join(HDF5_COLOR_TRAJ_PATH, floorplan_folder)
    if not os.path.isdir(h5_img_traj_floorplan_folder): os.mkdir(h5_img_traj_floorplan_folder)

    h5_img_ts_mask_floorplan_folder = os.path.join(HDF5_TRAJ_MASK_PATH, floorplan_folder)
    if not os.path.isdir(h5_img_ts_mask_floorplan_folder): os.mkdir(h5_img_ts_mask_floorplan_folder)

    h5_img_ts_and_id_mask_floorplan_folder = os.path.join(HDF5_TIME_AND_ID_MASK_PATH, floorplan_folder)
    if not os.path.isdir(h5_img_ts_and_id_mask_floorplan_folder): os.mkdir(h5_img_ts_and_id_mask_floorplan_folder)

    print(f'Drawing trajectories (RGB and float masks) into {floorplan_folder}...')

    for variation_image in os.listdir(os.path.join(HDF5_ROOT_FOLDER, floorplan_folder)):

        filename = os.path.join(HDF5_ROOT_FOLDER, floorplan_folder, variation_image)
        img = np.array(h5py.File(filename, 'r').get('img'))
        # plt.imshow(img, vmin=0, vmax=255)
   
        resolutions = img.shape[:2] # [800, 800]
        resolution_height, resolution_width  = resolutions[0], resolutions[1]

        trajectory_mask = np.zeros((resolution_width, resolution_height))
        trajectory_t_an_id_mask = np.zeros((resolution_width, resolution_height, 2))
        trajectory2timestamp = {}
        trajectory2timestamp_and_id = {}

        revit_img_path = os.path.join(REVIT_ROOT_FOLDER, floorplan_folder, f'img_variations_resolution_{resolution_height}_{resolution_width}')
        assert os.path.isdir(revit_img_path)

        # img = cv2.imread(floorplans_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 reads and writes in BGR fomat, not RGB

        # img_variations: either jpeg --> floorplans_path[i]/img_variations/*gt.jpeg or h5py --> base_plan/Revit/HDF5_IMAGES/folder[i]
        # crowdit_variations: within crowdit_simulations_path[i]

        endstrings = ['\n\r', '\r\n', '\r', '\n']
        csv_trajectory_file = os.path.join(crowdit_simulations_path, floorplan_folder, 'variation_'+variation_image.split('_')[-1].replace('.h5', ''))
        project_folder = 'project_'+floorplan_folder.split('__')[0]+'_'+variation_image.split('_')[-1].replace('.h5', '')+'_res'
        csv_location = os.path.join(csv_trajectory_file, project_folder, 'out', 'floor-'+floorplan_folder.split('__')[-1]+'_variation_'+variation_image.split('_')[-1].replace('.h5', '')+'.csv.gz')
        assert os.path.isfile(csv_location), f'csv path {csv_location} does not exist!'

        # Extract gzip file
        with gzip.open(csv_location, 'r') as f_in:
            # decode lines
            lines = [line.decode("utf-8").split(',') for line in f_in.readlines()]
            f_in.close()
        pedestrian_trajs = [
            [
                float(line[0]), # typecast time
                int(line[1]), # typecast pedId
                round(linear_interpolation(float(line[2]), min_x_dxf, max_x_dxf, 0, resolution_width)), # typecast x and project it to image resolution
                resolution_height - round(linear_interpolation(float(line[3]), min_y_dxf, max_y_dxf, 0, resolution_height)) # typecast y and project it to image resolution, also adjust origin offset in cv2
            ] for line in lines[1:]]

        # first sort after id, then after timestamp
        # pedestrian_trajs.sort(key=lambda x: (x[1], x[0]))
        pedestrian_trajs.sort(key=lambda x: (x[0], x[1]))

        max_time = max([ped[0] for ped in pedestrian_trajs])
        interval_length = 0.5
        num_intervals = int(max_time/interval_length)

        if maximum_overall_time < max_time: maximum_overall_time = max_time

        colors_by_timestamp = {i*interval_length: get_color_from_array(i, MAX_TIMESTAMPS) for i in range(1,num_intervals+1)}

        pedestrians_by_id = {}
        for ped in pedestrian_trajs:
            if ped[1]+1 in pedestrians_by_id:
                pedestrians_by_id[ped[1]+1].append([ped[0], ped[2], ped[3]])
            else:
                pedestrians_by_id.update({ped[1]+1: [[ped[0], ped[2], ped[3]]]})

        for pedId in pedestrians_by_id:
            for i in range(1, len(pedestrians_by_id[pedId])):
                x_start, y_start = pedestrians_by_id[pedId][i-1][1], pedestrians_by_id[pedId][i-1][2]
                x_end, y_end = pedestrians_by_id[pedId][i][1], pedestrians_by_id[pedId][i][2]
                
                # Create lines in RGB image
                color = colors_by_timestamp[pedestrians_by_id[pedId][i][0]]
                # cv2.line(img, (x_start, y_start), (x_end, y_end), (color), thickness=1) # cv2 origin: upper left
                
                # Create trajectory-timestamped image mask
                time_start, time_end = pedestrians_by_id[pedId][i-1][0], pedestrians_by_id[pedId][i][0]
                coord_line = list(zip(*line(*(x_start, y_start), *(x_end, y_end))))
                timestamp_line = [linear_interpolation(i, 0, len(coord_line), time_start, time_end) for i in range(len(coord_line))]
                # trajectory2timestamp.update({coord: timestamp for (coord, timestamp) in zip(coord_line, timestamp_line)})
                for (coord, timestamp) in zip(coord_line, timestamp_line):
                    if coord not in trajectory2timestamp_and_id:
                        # update only timestamp
                        trajectory2timestamp.update({coord: timestamp})
                        # update timestamp and pedestrian id
                        trajectory2timestamp_and_id.update({coord: [timestamp, pedId]})
                    else:
                        current_ts = trajectory2timestamp_and_id[coord][0]
                        if timestamp > trajectory2timestamp_and_id[coord][0]:
                            # update only timestamp
                            trajectory2timestamp.update({coord: timestamp})
                            # update timestamp and pedestrian id
                            trajectory2timestamp_and_id.update({coord: [timestamp, pedId]})
                        elif timestamp == trajectory2timestamp_and_id[coord][0]:
                            raise ValueError('Two trajectories cannot be at same place at the same time!')
                        else:
                            # if pedId's timestamp smaller and at the same place --> don't overwrite
                            continue

        # consistency check
        for key in trajectory2timestamp_and_id:
            assert trajectory2timestamp_and_id[key][0] == trajectory2timestamp[key]

        trajectory_coord_list = np.array(list(trajectory2timestamp.keys()))
        # Assign times to coordinates inside mask
        trajectory_mask[trajectory_coord_list[:,1], trajectory_coord_list[:,0]] = np.array(list(trajectory2timestamp.values()))

        # Assign times and ids to coordinates inside mask
        trajectory_t_an_id_mask[trajectory_coord_list[:,1], trajectory_coord_list[:,0]] = np.array(list(trajectory2timestamp_and_id.values()))
    
        # traj = trajectory_t_an_id_mask[:,:,1].squeeze()
        # non_zeros = np.argwhere(traj > 0)
        # for coord in non_zeros:
        #     id = traj[coord[0], coord[1]]
        #     color = get_color_from_pedId(id)
        #     # ts = traj[coord[0], coord[1]]
        #     #color = get_color_from_array(ts, MAX_TIME)
        #     img[coord[0], coord[1]] = color
        # plt.imshow(img, vmin=0, vmax=255)

        # Color-decoding the timestamps with respect to the maximum timestamp in the dataset
        # trajectory_mask_img = np.zeros((resolution_width, resolution_height, 3))
        # timestamp2color = {coord: get_color_from_array(trajectory2timestamp_and_id[coord][0], MAX_TIME) for coord in trajectory2timestamp_and_id}
        # for coord in timestamp2color:
        #     img[coord[1], coord[0]] = timestamp2color[coord]
        # plt.imshow(img, vmin=0, vmax=255)

        # cv2.imshow('Image', img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 reads and writes in BGR fomat, not RGB
        
        # plt.axis('off')
        # plt.imshow(img, vmin=0, vmax=255)

        # Store RGB images as JPEG
        # revit_img_variation_filepath = os.path.join(revit_img_path, variation_image.replace('HDF5_', '').replace('.h5', '_gt_traj.jpeg'))
        # plt.imsave(revit_img_variation_filepath, img, vmin=0, vmax=255)

        # Store RGB images as HDF5
        # h5_img_traj_filepath = os.path.join(h5_img_traj_floorplan_folder, variation_image)
        # hf_im = h5py.File(h5_img_traj_filepath, 'w')
        # hf_im.create_dataset('img', data=img)
        # hf_im.close()

        # Store trajectory masks images as HDF5
        # h5_img_ts_mask_filepath = os.path.join(h5_img_ts_mask_floorplan_folder, variation_image)
        # hf_mk = h5py.File(h5_img_ts_mask_filepath, 'w')
        # hf_mk.create_dataset('img', data=trajectory_mask)
        # hf_mk.close()

        # Store timestamp and id masks images as HDF5
        h5_img_ts_and_id_mask_filepath = os.path.join(h5_img_ts_and_id_mask_floorplan_folder, variation_image)
        hf_mk = h5py.File(h5_img_ts_and_id_mask_filepath, 'w')
        hf_mk.create_dataset('img', data=trajectory_t_an_id_mask)
        hf_mk.close()

        # Visualize stored hdf5 file
        # mask = np.array(h5py.File(h5_img_ts_mask_filepath, 'r').get('img'))
        # new_img = np.zeros((800,800,3))
        # non_zeros = np.argwhere(mask!=0)
        # for x,y in non_zeros:
        #     new_val = get_color_from_array(mask[x,y], MAX_TIME)
        #     new_img[x,y] = get_color_from_array(mask[x,y], MAX_TIME)
        # plt.imshow(new_img.astype('uint8'), vmin=0, vmax=255)
    

    # plt.plot(frs, gvals)
    # plt.show()

quit()

cmap = plt.cm.get_cmap('jet')
start, end = 65, 200
range = np.arange(start, end) # np.arange(cmap.N)
colors = cmap(range)
plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
plt.colorbar()
plt.show()

# fig, ax = plt.subplots()
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)

# im = ax.imshow(img, vmin=0, vmax=255)

# fig.colorbar(im, cax=cax, orientation='vertical')
# plt.show()

lines = lines
# ftype jarfile="C:\Users\ga78jem\Java\jdk1.8.0_131\jdk1.8.0_131\bin\javaw.exe" -jar -jar "%1" %*

# from matplotlib.colors import LinearSegmentedColormap
# def grayscale_cmap(cmap):
#     """Return a grayscale version of the given colormap"""
#     cmap = plt.cm.get_cmap(cmap)
#     colors = cmap(np.arange(cmap.N))
    
#     # convert RGBA to perceived grayscale luminance
#     # cf. http://alienryderflex.com/hsp.html
#     RGB_weight = [0.299, 0.587, 0.114]
#     luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
#     colors[:, :3] = luminance[:, np.newaxis]
        
#     return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
    
# def view_colormap(cmap):
#     """Plot a colormap with its grayscale equivalent"""
#     cmap = plt.cm.get_cmap(cmap)
#     colors = cmap(np.arange(65,200))
    
#     cmap = grayscale_cmap(cmap)
#     grayscale = cmap(np.arange(65,200))
    
#     fig, ax = plt.subplots(2, figsize=(6, 2),
#                            subplot_kw=dict(xticks=[], yticks=[]))
#     ax[0].imshow([colors], extent=[0, 10, 0, 1])
#     ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
# # view_colormap('jet')

# TWO POSSIBILITIES:
# one input image --> four output images (each correspoding to respective slice)
# one input image + one number {1,2,3,4} --> one image (corresponding to respective slice)