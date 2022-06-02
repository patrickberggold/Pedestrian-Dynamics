import argparse
import enum
import os
import xml.etree.ElementTree as ET
import numpy as np
import pylab as plt
from matplotlib.path import Path
import math
from requests import delete
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle
import h5py
from scipy import ndimage
from create_crowdit_projects import export2crowdit_project

def get_customized_colormap(class_names):

    len_classes = len(class_names.keys())
    
    viridis = cm.get_cmap('viridis', len_classes)
    custom_colors = viridis(np.linspace(0, 1, len_classes)) 
    # define colors
    for idx, key in enumerate(class_names):
        custom_colors[idx] = class_names[key]
    customed_colormap = ListedColormap(custom_colors)
    return customed_colormap

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

def point_plotter(points_dict):
    points = list(points_dict.keys())
    points = np.array(points)

    x = points[:,0]
    y = points[:,1]

    plt.scatter(x, y)
    plt.show()
    quit()

def line_plotter(line_list):
    for line in line_list:
        point1 = line[0]
        point2 = line[1]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        plt.plot(x_values, y_values, 'bo', linestyle="--")
        plt.text(point1[0]-0.015, point1[1]+0.25, f"({point1[0]}, {point1[1]})")
        plt.text(point2[0]-0.050, point2[1]-0.25, f"({point2[0]}, {point2[1]})")
    plt.show()

def check_num_values(line_list):
    sorted_by_val = {}
    for line in line_list:
        for coord in line:
            if coord not in sorted_by_val:
                sorted_by_val.update({coord: [1, [line]]})
            else:
                sorted_by_val[coord][0] += 1
                sorted_by_val[coord][1].append(line)
    return sorted_by_val

def preprocess_delete_short_lines(line_list):
    lines_per_point = check_num_values(line_list) # dict(key==coord: [num_lines, [lines_per_point]])
    preprocessed_line_list = []
    deleted_lines = []
    line_delete_count = 0
    for key in lines_per_point:
        assert lines_per_point[key][0] >= 2
        if lines_per_point[key][0] > 2:
            line_delete_count += 1
            assert lines_per_point[key][0] == 3
            line1 = lines_per_point[key][1][0]
            dist1 =  math.dist(line1[0], line1[1])

            line2 = lines_per_point[key][1][1]
            dist2 =  math.dist(line2[0], line2[1])
            
            line3 = lines_per_point[key][1][2]
            dist3 =  math.dist(line3[0], line3[1])

            if dist1 <= dist2 and dist1 <= dist2:
                if lines_per_point[key][1][0] not in deleted_lines:
                    deleted_lines.append(lines_per_point[key][1][0])
                del lines_per_point[key][1][0]
            elif dist2 <= dist1 and dist2 <= dist3:
                if lines_per_point[key][1][1] not in deleted_lines:
                    deleted_lines.append(lines_per_point[key][1][1])
                del lines_per_point[key][1][1]
            elif dist3 <= dist1 and dist3 <= dist2:
                if lines_per_point[key][1][2] not in deleted_lines:
                    deleted_lines.append(lines_per_point[key][1][2])
                del lines_per_point[key][1][2]
            else:
                raise ValueError('No smallest distance-> no line to delete in case of more than 2 lines...')

        for line in lines_per_point[key][1]:
            if line not in preprocessed_line_list:
                preprocessed_line_list.append(line)
            else:
                a =4
    
    return preprocessed_line_list, deleted_lines

# Input: current point coordinate, min floorplan coord, max floorplan coord, min resolution coord, max resolution coord -> output: projected coordinated
def linear_interpolation(curr_point_or, lim_min_or, lim_max_or, lim_min_proj, lim_max_proj):
    return lim_min_proj + (curr_point_or - lim_min_or) * (lim_max_proj - lim_min_proj) / (lim_max_or - lim_min_or)

def get_wall_lines_from_crowdit_xml(xml_filename):

    crowdit_filename = xml_filename.split('_')[0] + '.crowdit'

    crowdit_tree = ET.parse(crowdit_filename)
    
    origin_names = []
    destination_names = []

    for origin in crowdit_tree.findall('meta/morphosis/origin'):
        origin_names.append(origin.attrib['wunderZone'])
    
    for destination in crowdit_tree.findall('meta/morphosis/destination'):
        destination_names.append(destination.attrib['wunderZone'])

    tree = ET.parse(xml_filename)

    wall_points = []
    wall_lines = []
    crowdit_points = []
    crowdit_lines = []
    origin_lines = []
    origin_lines_projected = []
    destination_lines = []
    destination_lines_projected = []

    min_x = float("inf")
    max_x = -float("inf")
    min_y = float("inf")
    max_y = -float("inf")

    class_names = ['unpassable', 'walkable area', 'spawn_zone', 'destination']
    n_classes = len(class_names) + 1 # 0 is background

    for layer in tree.findall('layer'):
        if len(layer.findall('wall')) > 0 or len(layer.findall('wunderZone')) > 0:
            for wunderZone in layer.findall('wunderZone'):
                line = []
                if wunderZone.attrib['id'] in origin_names:
                    for point in wunderZone.findall('point'):
                        x = float(point.attrib['x'])
                        y = float(point.attrib['y'])
                        # crowdit_points.append((float(point.attrib['x']), float(point.attrib['y'])))
                        line.append((x, y))
                    origin_lines.append(line)
                elif wunderZone.attrib['id'] in destination_names:
                    for point in wunderZone.findall('point'):
                        x = float(point.attrib['x'])
                        y = float(point.attrib['y'])
                        # crowdit_points.append((float(point.attrib['x']), float(point.attrib['y'])))
                        line.append((x, y))
                    destination_lines.append(line)
            else: 
                for wall in layer.findall('wall'):
                    line = []
                    for point in wall.findall('point'):
                        x = float(point.attrib['x'])
                        y = float(point.attrib['y'])
                        if min_x > x:
                            min_x = x
                        if max_x < x:
                            max_x = x
                        if min_y > y:
                            min_y =y
                        if max_y < y:
                            max_y = y
                        # wall_points.append((x, y))
                        line.append((x, y))
                    wall_lines.append(line)
    
    return wall_lines, origin_lines, destination_lines, {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}

def get_wall_lines_from_pdf2xml_converter(xml_filename):

    tree = ET.parse(xml_filename)
    root = tree.getroot()

    wall_lines = []

    min_x = float("inf")
    max_x = -float("inf")
    min_y = float("inf")
    max_y = -float("inf")

    for child in root:
        for grandchild in child:
            if grandchild.tag == 'line':
                xyxy = grandchild.attrib['bbox'].split(',')
                line = [(float(xyxy[0]), float(xyxy[1])),(float(xyxy[2]), float(xyxy[3]))]
                for point in line:
                    x = float(point[0])
                    y = float(point[1])
                    if min_x > x:
                        min_x = x
                    if max_x < x:
                        max_x = x
                    if min_y > y:
                        min_y =y
                    if max_y < y:
                        max_y = y
                wall_lines.append(line)
                #print(grandchild.tag, grandchild.attrib)
        #print(child.tag, child.attrib)
    
    min_x_dxf = -0.15
    min_y_dxf = -0.15
    max_x_dxf = 20.15
    max_y_dxf = 20.15

    wall_lines_proj = []

    for line in wall_lines:
        line_proj = []
        for point in line:
            assert len(point) == 2
            x = point[0]
            y = point[1]
            x_proj = linear_interpolation(x, min_x, max_x, min_x_dxf, max_x_dxf)
            y_proj = linear_interpolation(y, min_y, max_y, min_y_dxf, max_y_dxf)
            line_proj.append((x_proj, y_proj))
        wall_lines_proj.append(line_proj)

    #return wall_lines_proj, origin_lines, destination_lines, {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}
    return wall_lines_proj, {'min_x': min_x_dxf, 'max_x': max_x_dxf, 'min_y': min_y_dxf, 'max_y': max_y_dxf}

def get_sorted_polynomial_point_list(line_list):
    sorted_points_list = [line_list[0][0], line_list[0][1]]
    del line_list[0]

    rectangle_points_list = []

    def point_not_in_list(list_to_query, line_to_check):
        if line_to_check not in list_to_query: return True
        else: return False # basically the closing point (if point already in the list)

    def get_next_point(list_to_query, startpoint):
        for line_idx, line in enumerate(list_to_query):
            for coord_idx, point in enumerate(line):
                if point[0] == startpoint[0] and point[1] == startpoint[1]:
                    del list_to_query[line_idx]
                    if coord_idx == 0:
                        return line[1]
                    elif coord_idx == 1:
                        return line[0]
                    else:
                        raise IndexError('coordinate index not in {0, 1}!')
        raise ValueError('Couldnt find successive point to given startpoint')

    while len(line_list) > 0:
        found_line = get_next_point(line_list, sorted_points_list[-1])
        if found_line and point_not_in_list(sorted_points_list, found_line):
            sorted_points_list.append(found_line)
        else:
            rectangle_points_list.append(sorted_points_list)
            if len(line_list) > 0:
                sorted_points_list = [line_list[0][0], line_list[0][1]]
                del line_list[0]

    return sorted_points_list

def create_input_images_and_crowdit_projects(xml_filename, root_dir, origins, destinations, class_names, resolutions):

    assert len(resolutions) == 2
    resolution_width, resolution_height = resolutions[0], resolutions[1]

    img_variations_dir = os.path.join('\\'.join(xml_filename.split('\\')[:-1]), f'img_variations_resolution_{resolution_width}_{resolution_height}')

    if not os.path.isdir(img_variations_dir):
        os.mkdir(img_variations_dir)

    # Separate GT segmented images directory
    array_gt_variations_dir = os.path.join(root_dir, f'..\\HDF5_GT_resolution_{resolution_width}_{resolution_height}')
    if not os.path.isdir(array_gt_variations_dir):
        os.mkdir(array_gt_variations_dir)

    # Separate flooplan images directory
    array_images_variations_dir = os.path.join(root_dir, f'..\\HDF5_IMAGES_resolution_{resolution_width}_{resolution_height}')
    if not os.path.isdir(array_images_variations_dir):
        os.mkdir(array_images_variations_dir)

    wall_lines, limits = get_wall_lines_from_pdf2xml_converter(xml_filename)

    min_x = limits['min_x']
    max_x = limits['max_x']
    min_y = limits['min_y']
    max_y = limits['max_y']

    # delete small, unnecessary lines between outside wall and interior
    preprocessed_lines, deleted_lines = preprocess_delete_short_lines(wall_lines)

    # export2crowdit_project(preprocessed_lines, origins, destinations, xml_filename.split('\\')[-2])

    # line_plotter(preprocessed_lines)
    # sort points to form a polygon
    wall_points_sorted = get_sorted_polynomial_point_list(preprocessed_lines) # exclude outer walls for now

    polygon = []
    # project wall corners to image resolution
    for id, point in enumerate(wall_points_sorted):
        # project floor plan dimensions to given resolution
        x_proj = int(linear_interpolation(float(point[0]), min_x, max_x, 0, resolution_width))
        y_proj = int(linear_interpolation(float(point[1]), min_y, max_y, 0, resolution_height))
        polygon.append((x_proj, y_proj))

    origin_rectangles_projected = []
    destination_rectangles_projected = []

    # project origins and destinations to image resolution
    for origin in origins:
        edges_proj = []
        for edge in origin:
            x = edge[0]
            y = edge[1]
            x_proj = int(linear_interpolation(x, min_x, max_x, 0, resolution_width))
            y_proj = int(linear_interpolation(y, min_y, max_y, 0, resolution_height))
            edges_proj.append((x_proj, y_proj))
        origin_rectangles_projected.append(edges_proj)
    
    for destination in destinations:
        edges_proj = []
        for edge in destination:
            x = edge[0]
            y = edge[1]
            x_proj = int(linear_interpolation(x, min_x, max_x, 0, resolution_width))
            y_proj = int(linear_interpolation(y, min_y, max_y, 0, resolution_height))
            edges_proj.append((x_proj, y_proj))
        destination_rectangles_projected.append(edges_proj)

    poly_path=Path(polygon)

    x, y = np.mgrid[:resolution_width, :resolution_height]
    coors = np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))

    mask = poly_path.contains_points(coors)
    base_mask = mask.reshape(resolution_width, resolution_height)
    base_mask = np.array(base_mask).astype(int)

    # plt.imshow(masked_img)

    base_image = np.zeros((resolution_width, resolution_height, 3))
    one_coords_base_image = np.argwhere(base_mask==1)
    base_image[one_coords_base_image[:,0], one_coords_base_image[:,1]] = np.array([255, 255, 255])

    for idx, origin in enumerate(origin_rectangles_projected):
        masked_img = base_mask.copy()
        image_array = base_image.copy()

        x_or_start, x_or_end = origin[0][0], origin[1][0]
        y_or_start, y_or_end = origin[0][1], origin[1][1]
       
        red_coords = np.array([(x_or, y_or) for x_or in range(x_or_start, x_or_end) for y_or in range(y_or_start, y_or_end)])

        image_array[red_coords[:,0], red_coords[:,1]] = np.array([255, 0, 0])
        masked_img[red_coords[:,0], red_coords[:,1]] = 2

        destination = destination_rectangles_projected[idx]

        x_dst_start, x_dst_end = destination[0][0], destination[1][0]
        y_dst_start, y_dst_end = destination[0][1], destination[1][1]

        green_coords = np.array([(x_dst, y_dst) for x_dst in range(x_dst_start, x_dst_end) for y_dst in range(y_dst_start, y_dst_end)])

        image_array[green_coords[:,0], green_coords[:,1]] = np.array([0, 255, 0])
        masked_img[green_coords[:,0], green_coords[:,1]] = 3

        image_array = ndimage.rotate(image_array, 90)
        masked_img = ndimage.rotate(masked_img, 90)

        # plt.axis('off')
        # plt.imshow(image_array, vmin=0, vmax=255)

    # for idx, origin in enumerate(origin_rectangles_projected):
        # destination = destination_rectangles_projected[idx]
        # for x_or in range(origin[0][0], origin[1][0]):
        #     for y_or in range(origin[0][1], origin[1][1]):
        #         masked_img[x_or,y_or] = 2
        #         image_array[x_or,y_or] = np.array([255, 0, 0])

        
        # for x_dst in range(destination[0][0], destination[1][0]):
        #     for y_dst in range(destination[0][1], destination[1][1]):
        #         masked_img[x_dst,y_dst] = 3
        #         image_array[x_dst,y_dst] = np.array([0, 255, 0])

        # plt.imshow(image_array, vmin=0, vmax=255)

        # plt.axis('off')
        #####################################################################################
        ################################### VISUALIZATION ###################################
        #####################################################################################
        # fig, ax = plt.subplots()
        # plt.axis('off')
        # ax.matshow(masked_img, cmap=get_customized_colormap(class_names))#plt.cm.Blues)

        # Logging
        cropped_print = xml_filename.split("\\")[-2].split("__")[0]
        print(f'Saving jpeg of floorplan {cropped_print}, variation {idx} ...')
        
        # save mask + colormap (ground truth)
        img_store_path = os.path.join(img_variations_dir, xml_filename.split('\\')[-1].replace('.xml', f'_variation_{idx}.jpeg'))
        plt.imsave(img_store_path.replace('.jpeg', '_gt_mask.jpeg'), masked_img, cmap=get_customized_colormap(class_names))
        # save RGB image (train data)
        plt.imsave(img_store_path.replace('.jpeg', '_input_img.jpeg'), image_array.astype(np.uint8), vmin=0, vmax=255)#
        
        # SAVE INPUT IMG ARRAY AS HDF5
        subgroup_folder = os.path.join(array_images_variations_dir, xml_filename.split('\\')[-2])
        if not os.path.isdir(subgroup_folder):
            os.mkdir(subgroup_folder)
        store_path = os.path.join(subgroup_folder, 'HDF5_'+xml_filename.split('\\')[-1].replace('.xml', f'_variation_{idx}.h5'))

        hf = h5py.File(store_path, 'w')
        hf.create_dataset('img', data=image_array.astype(np.uint8))
        hf.close()

        # SAVE HDF5 GROUND TRUTH DATA
        subgroup_folder = os.path.join(array_gt_variations_dir, xml_filename.split('\\')[-2])
        if not os.path.isdir(subgroup_folder):
            os.mkdir(subgroup_folder)
        store_path = os.path.join(subgroup_folder, 'HDF5_'+xml_filename.split('\\')[-1].replace('.xml', f'_variation_{idx}.h5'))

        # return mask, add 1 to include background (which needs to be 0)
        masked_img += 1

        hf = h5py.File(store_path, 'w')
        hf.create_dataset('img', data=masked_img)
        hf.close()

        # with open(os.path.join(subgroup_folder, 'classes.txt'), 'w') as fcl:
        #     fcl.write(f'0: background\n')
        #     for idx, key in enumerate(class_names):
        #         fcl.write(f'{idx+1}: {key}\n')
        # fcl.close()

        # plot saved hdf5 image
        # hf = h5py.File(store_path, 'r')
        # ds = hf.get('img')
        # ds = np.array(ds)
        # plt.imshow(ds, vmin=0, vmax=255)

        # Pickle
        # with open(os.path.join(array_variations_dir, 'PICKLE_'+xml_filename.split('\\')[-1].replace('.xml', f'_variation_{idx}.pkl')),'wb') as f: pickle.dump(masked_img, f)
        # f.close()

        # Numpy
        # np.save(os.path.join(array_variations_dir, 'NUMPY_'+xml_filename.split('\\')[-1].replace('.xml', f'_variation_{idx}.npy')), masked_img)


    # for i in range(15):
    #     for j in range(15):
    #         c = img[j,i]
    #         ax.text(i, j, str(c), va='center', ha='center')

    #plt.imshow(img)
    # plt.show()

    # quit()