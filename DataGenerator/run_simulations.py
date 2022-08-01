import gzip
import os
import re
from math import ceil
import shutil, errno

from numpy import array
from tqdm import tqdm

JAVA_EXE = 'C:\\Users\\ga78jem\\AppData\\Roaming\\accu-rate\\crowd-it\\bin\\java\\zulu11.50.19-ca-fx-jdk11.0.12-win_x64\\bin\\java.exe'
CROWDIT_KERNEL = '-jar C:\\Users\\ga78jem\\AppData\\Roaming\\accu-rate\\crowd-it\\bin\\kernel\\2.11.1.jar'
NUM_AGENTS = 10
AGENTS_SPAWNING_PER_SECOND = 2

missing_results_list = []
heavy_comput_list = []

def copy_dir(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

def manipulate_settings(crowdit_folder: str, num_agents: int = 40, spawn_endtime: int = 5):

    for layout_type in os.listdir(crowdit_folder):
        if layout_type != 'train_station':
            continue
        layout_folder = os.path.join(crowdit_folder, layout_type)
        crowdit_folder_list = sorted([dirname for dirname in os.listdir(layout_folder)], key=lambda x: int(x.split('__')[0]))
        for floorplan_folder in crowdit_folder_list:
            for variation_folder in os.listdir(os.path.join(layout_folder, floorplan_folder)):
                variation_folder_abs = os.path.join(layout_folder, floorplan_folder, variation_folder)
                num_agents_list = [int(ag_dir.split('_')[-1]) for ag_dir in os.listdir(variation_folder_abs) if not ag_dir.endswith('Kopie')]
                if num_agents not in num_agents_list:
                    # Create new folder structure
                    src_dir = os.path.join(variation_folder_abs, os.listdir(variation_folder_abs)[0])
                    new_ag_dir = os.path.join(layout_folder, floorplan_folder, variation_folder, f'AGENTS_PER_SRC_{str(num_agents)}')
                    copy_dir(src_dir, new_ag_dir)
                    project_folder = os.path.join(new_ag_dir, f'project_{floorplan_folder.split("_")[0]}_{variation_folder.split("_")[-1]}_res')
                    project_file = os.path.join(new_ag_dir, f'project_{floorplan_folder.split("_")[0]}_{variation_folder.split("_")[-1]}.crowdit')
                    
                    # renaming floorplan
                    floorplan_file = os.listdir(os.path.join(project_folder, 'geometry'))[0]
                    new_floorplan_file = '_'.join(floorplan_file.split('_')[:-1])+f'_{num_agents}.floor'
                    os.rename(os.path.join(project_folder, 'geometry', floorplan_file), os.path.join(project_folder, 'geometry', new_floorplan_file))
                    
                    # rewrite xml content in project file
                    with open(project_file, 'r') as f_r:
                        xml_content = f_r.read()
                    f_r.close()
                    exchange_string = re.search('geometry/(.*).floor', xml_content)
                    xml_content = xml_content.replace(exchange_string.group(1), new_floorplan_file.replace('.floor', ''))
                    with open(project_file, 'w') as f_w:
                        f_w.write(xml_content)
                    f_w.close()
                    
                    # TODO PROBLEM: dont know the spawn times because they depend on areas and num agents
                    # rewrite xml content in intervals file
                    intervals_xml_path = os.path.join(project_folder, 'meta', 'intervals.xml')
                    with open(intervals_xml_path, 'r') as f_r:
                        xml_content = f_r.read()
                    f_r.close()
                    exchange_string = re.search('how(.*)group=', xml_content)
                    xml_content = xml_content.replace(exchange_string.group(1), f'Many="{num_agents}" ')
                    with open(intervals_xml_path, 'w') as f_w:
                        f_w.write(xml_content)
                    f_w.close()

                else:
                    for num_agents_set in os.listdir(variation_folder_abs):
                        num_agents_folder = os.path.join(layout_folder, floorplan_folder, variation_folder, num_agents_set)
                        for project_folder in os.listdir(num_agents_folder):
                            if not os.path.isdir(os.path.join(num_agents_folder, project_folder)):
                                continue
                            intervals_xml_path = os.path.join(num_agents_folder, project_folder, 'meta', 'intervals.xml')
                            with open(intervals_xml_path, 'r') as f_r:
                                xml_content = f_r.read()
                            f_r.close()
                            # Manipulation
                            if num_agents and spawn_endtime:
                                assert 1 == 2
                                exchange_string = re.search('to=(.*)group=', xml_content)
                                xml_content = xml_content.replace(exchange_string.group(1), f'"{spawn_endtime}" howMany="{num_agents}" ')
                            elif not num_agents and spawn_endtime:
                                # determine the spawntime by defining the spawning agents per second
                                num_agents_ = int(num_agents_set.split('_')[-1])
                                spawing_agents_per_second = 2.0
                                spawn_endtime = ceil(num_agents_/spawing_agents_per_second)

                                exchange_string = re.search('to=(.*)group=', xml_content)
                                xml_content = xml_content.replace(exchange_string.group(1), f'"{spawn_endtime}" howMany="{num_agents_}" ')
                            with open(intervals_xml_path, 'w') as f_w:
                                f_w.write(xml_content)
                            f_w.close()
            print(f'Manipulating {floorplan_folder} with num_agents={num_agents}, spawning within {spawn_endtime} secs...')

def run_simulations(crowdit_folder: str, only_do: str):

    command = JAVA_EXE+' '+CROWDIT_KERNEL
    for layout_type in os.listdir(crowdit_folder):
        if layout_type != only_do:
            continue
        print('#####################################')
        print(f'START SIMULATING IN {layout_type}')
        print('#####################################')
        layout_folder = os.path.join(crowdit_folder, layout_type)
        crowdit_folder_list = sorted([dirname for dirname in os.listdir(layout_folder)], key=lambda x: int(x.split('__')[0]))
        for floorplan_folder in crowdit_folder_list:
            # if int(floorplan_folder.split('__')[0]) == 0:
            #     continue
            print(f'At floorplan {floorplan_folder}')
            variation_list = sorted([dirname for dirname in os.listdir(os.path.join(layout_folder, floorplan_folder))], key=lambda x: int(x.split('_')[1]))
            for variation_folder in tqdm(variation_list):
                for num_agents_set in os.listdir(os.path.join(layout_folder, floorplan_folder, variation_folder)):
                    num_agents_folder = os.path.join(layout_folder, floorplan_folder, variation_folder, num_agents_set)
                    for project_file in os.listdir(num_agents_folder):
                        if not project_file.endswith('.crowdit'):
                            continue
                        setting = f'--filename={os.path.join(layout_folder, floorplan_folder, variation_folder, num_agents_set, project_file)} --lastStep=100000'
                        # 'C:\\Users\\ga78jem\\Documents\\Crowdit\\0__floorplan_zPos_0.0_roomWidth_0.24_numRleft_2.0_numRright_2.0\\variation_1\\project_0_1.crowdit --lastStep=180'
                        print(f'\n\nSimulating {project_file}, num_agents {num_agents_set.split("_")[-1]}...')
                        os.system(command+' '+setting)

def check_simulation_runtimes(crowdit_folder: str, only_do: str):
    for layout_type in os.listdir(crowdit_folder):
        if layout_type != only_do:
            continue
        print('#####################################')
        print(f'START INSPECTING IN {layout_type}')
        print('#####################################')
        layout_folder = os.path.join(crowdit_folder, layout_type)
        crowdit_folder_list = sorted([dirname for dirname in os.listdir(layout_folder)], key=lambda x: int(x.split('__')[0]))
        for floorplan_folder in crowdit_folder_list:
            # if int(floorplan_folder.split('__')[0]) == 0:
            #     continue
            print(f'At floorplan {floorplan_folder}')
            variation_list = sorted([dirname for dirname in os.listdir(os.path.join(layout_folder, floorplan_folder))], key=lambda x: int(x.split('_')[1]))
            for variation_folder in tqdm(variation_list):
                for num_agents_set in os.listdir(os.path.join(layout_folder, floorplan_folder, variation_folder)):
                    num_agents_folder = os.path.join(layout_folder, floorplan_folder, variation_folder, num_agents_set)
                    for project_file in os.listdir(num_agents_folder):
                        if not project_file.endswith('.crowdit'):
                            project_folder = os.path.join(num_agents_folder, project_file)
                            if not os.path.isdir(os.path.join(project_folder, 'out')):
                                missing_results_list.append(os.path.join(layout_type, floorplan_folder, variation_folder, num_agents_set))
                            else:
                                csv_filename = [filename for filename in os.listdir(os.path.join(project_folder, 'out')) if filename.endswith('.gz')]
                                csv_filename = csv_filename[0] if len(csv_filename) > 0 else None
                                if not csv_filename:
                                    he = 1
                                csv_location_path = os.path.join(project_folder, 'out', csv_filename)
                                with gzip.open(csv_location_path, 'r') as f_in:
                                    # decode lines
                                    lines = [line.decode("utf-8").split(',') for line in f_in.readlines()]
                                    f_in.close()
                                times = array([float(line[0]) for line in lines[1:]])
                                max_time = max(times)
                                if max_time > 120:
                                    heavy_comput_list.append(os.path.join(layout_type, floorplan_folder, variation_folder, num_agents_set))

# check_simulation_runtimes('C:\\Users\\ga78jem\\Documents\\Crowdit\\ADVANCED_EXPERIMENTS', 'train_station')
# manipulate_settings('C:\\Users\\ga78jem\\Documents\\Crowdit\\ADVANCED_EXPERIMENTS', num_agents=35, spawn_endtime=None)
# run_simulations('C:\\Users\\ga78jem\\Documents\\Crowdit\\ADVANCED_EXPERIMENTS', 'train_station')

debug_stop = True