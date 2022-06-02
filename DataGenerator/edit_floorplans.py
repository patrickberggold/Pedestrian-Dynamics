import sys
import ezdxf
from ezdxf.acc.vector import Vec3
import argparse
import os
import random
from pdf2xml import convert_pdf_to_xml
from create_floorplan_images import create_input_images_and_crowdit_projects
import numpy as np
import itertools

# parser = argparse.ArgumentParser()
# parser.add_argument("--foldername", type=str)
# args = parser.parse_args()

def create_ors_dsts_and_export_dxf(dxf_filepath, txt_filepath, export_dxf: bool = False):

    # Create Crowdit folder
    crowdit_folderpath = os.path.join(dxf_filepath.split('Revit')[0], 'Crowdit')
    if not os.path.isdir(crowdit_folderpath): os.mkdir(crowdit_folderpath)

    # Create subfolder for each floorplan base
    crowdit_floorplan_subpath = os.path.join(crowdit_folderpath, dxf_filepath.split('Exports')[-1].split('\\')[1])
    if not os.path.isdir(crowdit_floorplan_subpath): os.mkdir(crowdit_floorplan_subpath)

    dxf_variations_dir = os.path.join('\\'.join(dxf_filepath.split('\\')[:-1]), 'dxf_variations')

    if not os.path.isdir(dxf_variations_dir):
        os.mkdir(dxf_variations_dir)

    rooms = []
    room = []

    num_origins = 1
    num_destinations = 1

    z_value_string = dxf_filepath.split('\\')[-1].split('zPos_')[-1].split('_')[0]
    if z_value_string=='0.0' or z_value_string=='0': 
        z_wall_there = True

    with open(txt_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.endswith('\n'): line = line.strip().replace('\n', '')
            if line.startswith('ROOM_'):
                if len(room) > 0:
                    rooms.append(room)
                    room = []
            elif line.startswith('COORD_'):
                x,y = line.split('COORD_')[-1].split(',')
                coord = (float(x), float(y))
                room.append(coord)
        if len(room) > 0:
            rooms.append(room)
    f.close()

    # Sort rooms list by room coordinates (first x, then y)
    sorted_rooms_by_x = sorted(rooms, key=lambda element: (element[0][0], element[0][1]), reverse=False)
    sorted_rooms_dict = {f'room_{i}': room_coords for i, room_coords in enumerate(sorted_rooms_by_x)}
    permuted_rooms = [combi for combi in itertools.permutations(sorted_rooms_dict, num_origins+num_destinations)]

    origin_coords = []
    destination_coords = []

    for idx, room_comb in enumerate(permuted_rooms):

        origin_room = sorted_rooms_dict[room_comb[0]]
        destination_room = sorted_rooms_dict[room_comb[1]]

        #origin_room, destination_room = random.sample(rooms, num_origins + num_destinations)
        distance_to_borders = 0.5
        s1 = origin_room[0][0] + distance_to_borders
        s2 = origin_room[0][1] + distance_to_borders
        s3 = origin_room[1][0] - distance_to_borders
        s4 = origin_room[1][1] - distance_to_borders
        assert s3 > s1+0.5
        assert s4 > s2+0.5

        d1 = destination_room[0][0] + distance_to_borders
        d2 = destination_room[0][1] + distance_to_borders
        d3 = destination_room[1][0] - distance_to_borders
        d4 = destination_room[1][1] - distance_to_borders
        assert d3 > d1+0.5
        assert d4 > d2+0.5

        origin_coords.append([(s1, s2), (s3, s4)])
        destination_coords.append([(d1, d2), (d3, d4)])

        if export_dxf:

            try:
                doc = ezdxf.readfile(dxf_filepath)
            except IOError:
                print(f"Not a DXF file or a generic I/O error.")
                sys.exit(1)
            except ezdxf.DXFStructureError:
                print(f"Invalid or corrupted DXF file.")
                sys.exit(2)

            doc.layers.add(name="crowdit", color=7, linetype="DASHED")
            model_space = doc.modelspace()

            # "0.9485693795314933,2.974038089426853,4.323054061934929,0.6475300164654794"
            or_polyline = model_space.add_polyline3d([
                Vec3(s1, s2, 0.),
                Vec3(s3, s2, 0.),
                Vec3(s3, s4, 0.),
                Vec3(s1, s4, 0.)],
                close=True,
                dxfattribs={"layer": "crowdit"}
            )
            # "16.1,19.3,19.5,15.5"
            dst_polyline = model_space.add_polyline3d([
                Vec3(d1, d2, 0.),
                Vec3(d3, d2, 0.),
                Vec3(d3, d4, 0.),
                Vec3(d1, d4, 0.)],
                close=True,
                dxfattribs={"layer": "crowdit"}
            )

            # remove unnecessary layers so crowdit works
            layer_removal_list = [layer.dxf.name for layer in doc.layers if layer.dxf.name=='Defpoints']# layer.dxf.name != 'crowdit' and not layer.dxf.name.startswith('A-WALL')]        
            for remove_layer in layer_removal_list:
                doc.layers.remove(remove_layer)

            store_path = os.path.join(dxf_variations_dir, dxf_filepath.split('\\')[-1].replace('.dxf', f'_variation_{idx}.dxf'))
            doc.saveas(store_path)

            crowdit_variation_subpath = os.path.join(crowdit_floorplan_subpath, f'variation_{idx}')
            if not os.path.isdir(crowdit_variation_subpath): os.mkdir(crowdit_variation_subpath)

        #doc.saveas(dxf_filepath.replace('.dxf', f'_py_edited_{idx}.dxf'))
        # del or_polyline.vertices[:4]
        # del dst_polyline.vertices[:4]
        # del or_polyline
        # del dst_polyline
        
    return origin_coords, destination_coords

class_names = {
    # get color values from https://doc.instantreality.org/tools/color_calculator/
    # Format RGBA
    'unpassable': np.array([0, 0, 0, 1]), 
    'walkable area': np.array([1, 1, 1, 1]), 
    'spawn_zone': np.array([0.501, 0.039, 0, 1]), #np.array([1, 0.0, 0, 1]) 
    'destination': np.array([0.082, 0.847, 0.054, 1]), # np.array([0.0, 1, 0, 1]), 
    # 'path': np.array([0, 0, 1, 1])
    # start: [91, 235, 251] -> [91, 96, 251]
    }

REVIT_PATH = "C:\\Users\\ga78jem\\Documents\\Revit\\Exports\\"

for idx, dir_name in enumerate(os.listdir(REVIT_PATH)):

    print(f'\n\nSTEP [{idx+1}/{len(os.listdir(REVIT_PATH))}]\n\n')

    filepath = os.path.join(REVIT_PATH, dir_name)

    txt_filepath = os.path.join(filepath, dir_name.split('__')[-1]+'.txt')
    dxf_filepath = os.path.join(filepath, dir_name.split('__')[-1]+'.dxf')
    pdf_filepath = os.path.join(filepath, dir_name.split('__')[-1]+'.pdf')

    txt_true = os.path.isfile(txt_filepath)
    pdf_true = os.path.isfile(pdf_filepath)
    dxf_true = os.path.isfile(dxf_filepath)

    xml_path = convert_pdf_to_xml(pdf_filepath)

    origins, destinations = create_ors_dsts_and_export_dxf(dxf_filepath, txt_filepath)

    resolutions = [800, 800]
    create_input_images_and_crowdit_projects(xml_path, REVIT_PATH, origins, destinations, class_names, resolutions)