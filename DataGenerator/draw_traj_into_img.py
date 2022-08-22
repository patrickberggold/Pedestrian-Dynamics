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
from run_simulations import manipulate_settings, run_simulations
from tqdm import tqdm

CROWDIT_INTERVAL_LENGTH = 0.5

CREATE_MASKS = True
CREATE_CSV = True

# manipulate_settings('C:\\Users\\ga78jem\\Documents\\Crowdit', num_agents=NUM_AGENTS, spawn_endtime=5)
# run_simulations('C:\\Users\\ga78jem\\Documents\\Crowdit')

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

BASE_PATH = "C:\\Users\\ga78jem\\Documents\\"

FLOORPLANS_GT_PATH = os.path.join(BASE_PATH, "Revit\\ADVANCED_FLOORPLANS")
CROWDIT_PATH = os.path.join(BASE_PATH, "Crowdit\\ADVANCED_EXPERIMENTS")

HDF5_ROOT_FOLDER = 'C:\\Users\\ga78jem\\Documents\\Revit\\DATASET_IMGs_1_SRC__1_DST\\HDF5_INPUT_IMAGES_resolution_800_800'
REVIT_ROOT_FOLDER = 'C:\\Users\\ga78jem\\Documents\\Revit\\Exports'

HDF5_TRAJ_MASK_PATH = os.path.join(BASE_PATH, "Revit\\ADVANCED_FLOORPLANS\\HDF5_GT_TIMESTAMP_MASKS_thickness_5")
if not os.path.isdir(HDF5_TRAJ_MASK_PATH): os.mkdir(HDF5_TRAJ_MASK_PATH)

CSV_SIMULATION_PATH = os.path.join(BASE_PATH, "Revit\\ADVANCED_FLOORPLANS\\CSV_GT_TRAJECTORIES")
if not os.path.isdir(CSV_SIMULATION_PATH): os.mkdir(CSV_SIMULATION_PATH)

for layout_type in os.listdir(FLOORPLANS_GT_PATH):
    if layout_type != 'train_station':
        continue
    layout_folder = os.path.join(FLOORPLANS_GT_PATH, layout_type)
    # Create new target head folder for the layout type in traj mask folder
    if not os.path.isdir(os.path.join(HDF5_TRAJ_MASK_PATH, layout_type)): os.mkdir(os.path.join(HDF5_TRAJ_MASK_PATH, layout_type))

    # Create new target head folder for the layout type in csv folder
    if not os.path.isdir(os.path.join(CSV_SIMULATION_PATH, layout_type)): os.mkdir(os.path.join(CSV_SIMULATION_PATH, layout_type))

    for floorplan_folder in os.listdir(layout_folder):

        # Create new target folder for each floorplan in traj mask folder
        if not os.path.isdir(os.path.join(HDF5_TRAJ_MASK_PATH, layout_type, floorplan_folder)): os.mkdir(os.path.join(HDF5_TRAJ_MASK_PATH, layout_type, floorplan_folder))
        
        # Create new target folder for each floorplan in csv folder
        if not os.path.isdir(os.path.join(CSV_SIMULATION_PATH, layout_type, floorplan_folder)): os.mkdir(os.path.join(CSV_SIMULATION_PATH, layout_type, floorplan_folder))
        
        print(f'Drawing trajectories (RGB and float masks) into {layout_type}, {floorplan_folder}...')

        # determine min & max values in original coordinate system (which is dxf)
        dxf_filepath = os.path.join(BASE_PATH, 'Revit\\Exports_ADVANCED', layout_type, floorplan_folder, floorplan_folder.split("__")[-1]+'.dxf')
        with open(dxf_filepath, 'r') as f_dxf:
            dxf_lines = f_dxf.readlines()
            min_id = dxf_lines.index('$EXTMIN\n')
            max_id = dxf_lines.index('$EXTMAX\n')
            min_x_dxf = float(dxf_lines[min_id+2])
            min_y_dxf = float(dxf_lines[min_id+4])
            max_x_dxf = float(dxf_lines[max_id+2])
            max_y_dxf = float(dxf_lines[max_id+4])
        f_dxf.close()
        sizes = {'x': [min_x_dxf, max_x_dxf], 'y': [min_y_dxf, max_y_dxf]}

        max_time_per_flooplan = 0.

        for variation_image in tqdm(os.listdir(os.path.join(layout_folder, floorplan_folder))):

            # Create new target folder for each src-dst variation in traj mask folder
            target_variation_folder_mask = os.path.join(HDF5_TRAJ_MASK_PATH, layout_type, floorplan_folder, f'variation_{variation_image.split("_")[-1].replace(".h5", "")}')
            if not os.path.isdir(target_variation_folder_mask) and CREATE_MASKS: os.mkdir(target_variation_folder_mask)
            
            # Create new target folder for each src-dst variation in csv folder
            target_variation_folder_csv = os.path.join(CSV_SIMULATION_PATH, layout_type, floorplan_folder, f'variation_{variation_image.split("_")[-1].replace(".h5", "")}')
            if not os.path.isdir(target_variation_folder_csv) and CREATE_CSV: os.mkdir(target_variation_folder_csv)
            
            filename = os.path.join(layout_folder, floorplan_folder, variation_image)
            img = np.array(h5py.File(filename, 'r').get('img'))
            # plt.imshow(img)

            resolutions = img.shape[:2] # [800, 800]
            resolution_height, resolution_width  = resolutions[0], resolutions[1]

            trajectory_mask = np.zeros((resolution_height, resolution_width))

            trajectory2timestamp = {}

            endstrings = ['\n\r', '\r\n', '\r', '\n']
            csv_trajectory_folder = os.path.join(CROWDIT_PATH, layout_type, floorplan_folder.replace('floorplan_', ''), 'variation_'+variation_image.split('_')[-1].replace('.h5', ''))
            for num_agent_folder in os.listdir(csv_trajectory_folder):

                # Create new target folder for each src-dst variation in traj mask folder
                target_filepath_mask = os.path.join(target_variation_folder_mask, f'variation_{variation_image.split("_")[-1].replace(".h5", "")}_num_agents_{num_agent_folder.split("_")[-1]}.h5')
                
                # Create new target folder for each src-dst variation in csv folder
                target_filepath_csv = os.path.join(target_variation_folder_csv, f'variation_{variation_image.split("_")[-1].replace(".h5", "")}_num_agents_{num_agent_folder.split("_")[-1]}.txt')
            
                project_folder = 'project_'+floorplan_folder.split('__')[0]+'_'+variation_image.split('_')[-1].replace('.h5', '')+'_res'
                csv_location_path = os.path.join(csv_trajectory_folder, num_agent_folder, project_folder, \
                    'out', f'floor-floorplan_variation_{variation_image.split("_")[-1].replace(".h5", "")}_agents_{num_agent_folder.split("_")[-1]}.csv.gz')#+ \
                        # floorplan_folder.split('__')[-1]+'_variation_'+variation_image.split('_')[-1].replace('.h5', '')+'.csv.gz')
                assert os.path.isfile(csv_location_path), f'csv path {csv_location_path} does not exist!'

                # Extract gzip file
                with gzip.open(csv_location_path, 'r') as f_in:
                    # decode lines
                    lines = [line.decode("utf-8").split(',') for line in f_in.readlines()]
                    f_in.close()

                if CREATE_CSV:
                    with open(target_filepath_csv, 'w') as f:
                        for line_el in lines[1:]:
                            line_el[-1] = line_el[-1].split('\r')[0]
                            line_string = ','.join(line_el)+'\n'
                            f.write(line_string)

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

                max_timestamp_per_variation = max([ped[0] for ped in pedestrian_trajs])
                if max_timestamp_per_variation > max_time_per_flooplan:
                    max_time_per_flooplan = max_timestamp_per_variation

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
                        # color = colors_by_timestamp[pedestrians_by_id[pedId][i][0]]
                        # cv2.line(img, (x_start, y_start), (x_end, y_end), (color), thickness=1) # cv2 origin: upper left
                        def linemaker(p_start, p_end, thickness=1):
                            x_start, x_end, y_start, y_end = p_start[0], p_end[0], p_start[1], p_end[1]
                            
                            x_diff, y_diff = x_end - x_start, y_end -y_start
                            m = y_diff / x_diff if x_diff != 0 else 10.

                            lines = []
                            or_line = [coord for coord in zip(*line(*(p_start[0], p_start[1]), *(p_end[0], p_end[1])))]
                            lines += [[coord] for coord in zip(*line(*(p_start[0], p_start[1]), *(p_end[0], p_end[1])))]

                            line_factors = []
                            for i in range(thickness-1):
                                sign = 1
                                if i%2 != 0:
                                    sign = -1
                                i = int(i/2.)+1
                                # th=2: +1, th=3: [+1,-1], th=4: [+1,-1,+2]
                                line_factors.append(sign*i)

                            for factor in line_factors:

                                if abs(m) > 1:
                                    extra_line = list(zip(*line(*(p_start[0]+factor, p_start[1]), *(p_end[0]+factor, p_end[1]))))
                                    # extra_line = list(zip(*_line_profile_coordinates((x_start+1, y_start), (x_end+1, y_end), linewidth=1)))
                                else:
                                    extra_line = list(zip(*line(*(p_start[0], p_start[1]+factor), *(p_end[0], p_end[1]+factor))))
                                    # extra_line = list(zip(*_line_profile_coordinates((x_start, y_start+1), (x_end, y_end+1), linewidth=1)))
                                # lines += extra_line
                                for idx in range(len(lines)):
                                    lines[idx].append(extra_line[idx])

                                # check if all points are offsetted correctly
                                for c_line_or, c_line_ex in zip(or_line, extra_line):
                                    if sum(c_line_ex) != sum(c_line_or)+factor:
                                        hi = 1

                            return lines
                        
                        # Create trajectory-timestamped image mask
                        time_start, time_end = pedestrians_by_id[pedId][i-1][0], pedestrians_by_id[pedId][i][0]

                        coord_lines = linemaker((x_start, y_start), (x_end, y_end), thickness=5)
                        
                        timestamp_line = [linear_interpolation(i, 0, len(coord_lines), time_start, time_end) for i in range(len(coord_lines))]

                        for (coords, timestamp) in zip(coord_lines, timestamp_line):
                            for coord in coords:
                                if coord not in trajectory2timestamp:
                                    # update only timestamp
                                    trajectory2timestamp.update({coord: (timestamp, 1.)})
                                else:
                                    current_ts_avg = trajectory2timestamp[coord][0]
                                    current_ts_counter = trajectory2timestamp[coord][1]
                                    trajectory2timestamp[coord] = (
                                        current_ts_avg + timestamp/(current_ts_counter + 1.) - current_ts_avg/(current_ts_counter + 1.),
                                        current_ts_counter + 1
                                    )
                                    # new_ts_avg = trajectory2timestamp[coord][0]
                                    # new_ts_counter = trajectory2timestamp[coord][1]


                trajectory_coord_list = np.array(list(trajectory2timestamp.keys()))
                # Assign times to coordinates inside mask
                trajectory_mask[trajectory_coord_list[:,1], trajectory_coord_list[:,0]] = np.array(list(trajectory2timestamp.values()))[:,0]

                # plt.imshow(trajectory_mask)

                # traj = trajectory_mask
                # non_zeros = np.argwhere(traj > 0)
                # for coord in non_zeros:
                #     # id = traj[coord[0], coord[1]]
                #     # color = get_color_from_pedId(id)
                #     ts = traj[coord[0], coord[1]]
                #     color = get_color_from_array(ts, MAX_TIME)
                #     img[coord[0], coord[1]] = color
                # plt.imshow(img, vmin=0, vmax=255)

                if CREATE_MASKS:
                    # Store trajectory masks images as HDF5
                    hf_mk = h5py.File(target_filepath_mask, 'w')
                    hf_mk.create_dataset('img', data=trajectory_mask)
                    hf_mk.create_dataset('max_time', data=max_timestamp_per_variation)
                    hf_mk.close()

        print(f'max time in {layout_type}, {floorplan_folder}: {max_time_per_flooplan:.3f}')