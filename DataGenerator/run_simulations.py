import os
import re
from subprocess import check_output
import subprocess

JAVA_EXE = 'C:\\Users\\ga78jem\\AppData\\Roaming\\accu-rate\\crowd-it\\bin\\java\\zulu11.50.19-ca-fx-jdk11.0.12-win_x64\\bin\\java.exe'
CROWDIT_KERNEL = '-jar C:\\Users\\ga78jem\\AppData\\Roaming\\accu-rate\\crowd-it\\bin\\kernel\\2.11.1.jar'

def manipulate_settings(crowdit_folder: str, num_agents: int = 40, spawn_endtime: int = 5):
    crowdit_folder_list = sorted([dirname for dirname in os.listdir(crowdit_folder) if '__floorplan' in dirname], key=lambda x: int(x.split('__')[0]))
    for floorplan_folder in crowdit_folder_list:
        for variation_folder in os.listdir(os.path.join(crowdit_folder, floorplan_folder)):
            variation_folder_counter = 0
            for project_folder in os.listdir(os.path.join(crowdit_folder, floorplan_folder, variation_folder)):
                if not os.path.isdir(os.path.join(os.path.join(crowdit_folder, floorplan_folder, variation_folder), project_folder)):
                    continue
                variation_folder_counter += 1
                intervals_xml_path = os.path.join(crowdit_folder, floorplan_folder, variation_folder, project_folder, 'meta', 'intervals.xml')
                with open(intervals_xml_path, 'r') as f_r:
                    xml_content = f_r.read()
                f_r.close()
                # Manipulation
                exchange_string = re.search('to=(.*)group=', xml_content)
                xml_content = xml_content.replace(exchange_string.group(1), f'"{spawn_endtime}" howMany="{num_agents}" ')
                with open(intervals_xml_path, 'w') as f_w:
                    f_w.write(xml_content)
                f_w.close()
            assert variation_folder_counter == 1
        print(f'Manipulating {floorplan_folder} with num_agents={num_agents}, spawning within {spawn_endtime} secs...')

def run_simulations(crowdit_folder: str):

    command = JAVA_EXE+' '+CROWDIT_KERNEL

    crowdit_folder_list = sorted([dirname for dirname in os.listdir(crowdit_folder) if '__floorplan' in dirname], key=lambda x: int(x.split('__')[0]))
    for floorplan_folder in crowdit_folder_list:
        for variation_folder in os.listdir(os.path.join(crowdit_folder, floorplan_folder)):
            variation_folder_counter = 0
            for project_file in os.listdir(os.path.join(crowdit_folder, floorplan_folder, variation_folder)):
                if not project_file.endswith('.crowdit'):
                    continue
                setting = f'--filename={os.path.join(crowdit_folder, floorplan_folder, variation_folder, project_file)} --lastStep=1000'
                # 'C:\\Users\\ga78jem\\Documents\\Crowdit\\0__floorplan_zPos_0.0_roomWidth_0.24_numRleft_2.0_numRright_2.0\\variation_1\\project_0_1.crowdit --lastStep=180'
                print(f'\n\nSimulating {project_file}...')
                os.system(command+' '+setting)

manipulate_settings('C:\\Users\\ga78jem\\Documents\\Crowdit', num_agents=40, spawn_endtime=5)
run_simulations('C:\\Users\\ga78jem\\Documents\\Crowdit')

# def run_win_cmd(cmd):
    
#     os.system(cmd)
#     hi = 2
#     # process = subprocess.Popen(cmd,
#     #                            shell=True,
#     #                            stdout=subprocess.PIPE,
#     #                            stderr=subprocess.PIPE)
#     # # stdout, stderr = process.communicate()
#     # stdout = process.stdout
#     # result = [line for line in process.stdout]
#     # # result = [line.decode('ISO-8859-1') for line in process.stdout if line.decode('ISO-8859-1') != '\r\n']
#     # errcode = process.returncode
#     # for line in result:
#     #     print(line)
#     # if errcode is not None:
#     #     raise Exception('cmd %s failed, see above for details', cmd)


# run_win_cmd(JAVA_EXE+' '+CROWDIT_KERNEL+' '+SETTINGS)
# # run_win_cmd('dir')

# # Manipulate numAgents --> change in intervals.xml  --> then run java kernal via command line