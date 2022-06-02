import os

def write_crowdit_meta_folder(meta_folder):

    intervals_text = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<root xmlFormat="0.10.2" isoDate="2022-05-25T09:02:45.691517600Z">\n\t<intervals id="Quelle-0_INTERVALS">\n\t\t<interval from="0" to="5" howMany="20" group="false"/>\n\t</intervals>\n\t<intervals id="Ziel-0_INTERVALS"/>\n</root>'
    intervals_file = os.path.join(meta_folder, 'intervals.xml')

    with open(intervals_file, 'w') as f_i:
        f_i.write(intervals_text)
    f_i.close()

    personas_text = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<agents xmlFormat="0.10.2" isoDate="2022-05-25T09:02:45.686517300Z">\n\t<persona id="DEFAULT">\n\t\t<velocity>\n\t\t\t<distribution min="0.46" max="1.61" mean="1.34" deviation="0.26" type="normal"/>\n\t\t</velocity>\n\t\t<torsoSize acceptedOverlap="0.15">\n\t\t\t<distribution min="0.42" max="0.46" mean="0.44" deviation="0.0" type="uniform"/>\n\t\t</torsoSize>\n\t\t<behavior perceptionRadius="2.0" socialDist="0.0"/>\n\t\t<ignoredSimulationObjects/>\n\t</persona>\n</agents>'
    personas_file = os.path.join(meta_folder, 'personas.xml')

    with open(personas_file, 'w') as f_p:
        f_p.write(personas_text)
    f_p.close()

    premove_times_text = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<root xmlFormat="0.10.2" isoDate="2022-05-25T09:02:45.693517400Z">\n\t<premovement allAtOnce="false" id="Quelle-0_PREMOVE">\n\t\t<distribution min="0.0" max="0.0" mean="0.0" deviation="0.0" type="none"/>\n\t</premovement>\n</root>'
    premove_times_file = os.path.join(meta_folder, 'premovement-times.xml')

    with open(premove_times_file, 'w') as f_pt:
        f_pt.write(premove_times_text)
    f_pt.close()

def write_crowdit_project_file(project_filename, project_name, floor_filename):

    xml_header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    scenario_header = f'<scenario xmlFormat="0.10.2" isoDate="2022-05-25T09:02:45.684518300Z" name="{project_name}">\n\t<spatial>\n'
    floor_line = f'\t\t<floor id="{floor_filename.split("/")[1].replace(".floor", "")}" floorAt="{floor_filename}" />\n'    
    scenario_bottom = '\t</spatial>\n\t<meta>\n\t\t<paths>\n\t\t\t<path id="Pfad-0" ratio="1.0">Quelle-0,Ziel-0</path>\n\t\t</paths>\n\t\t<backgroundImages />\n\t\t<pedCharacters>\n\t\t\t<type name="FPLMWalking" ratio="1.0" />\n\t\t</pedCharacters>\n\t\t<morphosis>\n'
    origin_head = '\t\t\t<origin id="Quelle-0" wunderZone="simObj-192">\n'
    origin_bottom = '\t\t\t\t<personas>\n\t\t\t\t\t<persona ref="DEFAULT">1.0</persona>\n\t\t\t\t</personas>\n\t\t\t\t<premovement ref="Quelle-0_PREMOVE" />\n\t\t\t\t<positioning generationPattern="quadratic" comfortDistance="0.0" />\n\t\t\t\t<intervals ref="Quelle-0_INTERVALS" />\n\t\t\t</origin>\n'
    destination_lines = '\t\t\t<destination id="Ziel-0" wunderZone="simObj-198" disableDynamicFlooding="false">\n\t\t\t\t<intervals ref="Ziel-0_INTERVALS" />\n\t\t\t</destination>\n'
    filler = '\t\t</morphosis>\n\t\t<sets />\n\t\t<pathSnippets />\n\t\t<floorProps>\n'
    floorProps = f'\t\t\t<floorProp floor="{floor_filename.split("/")[1].replace(".floor", "")}" cellDist="0.1" elevation="0.0" height="2.5" />\n\t\t</floorProps>\n'
    filler2 = '\t\t<groups>\n\t\t\t<row>1,1.0</row>\n\t\t</groups>\n\t\t<elevatorMatrices />\n\t</meta>\n'
    bottom = '\t<settings />\n\t<evaluations>\n\t\t<measurements />\n\t</evaluations>\n\t<visualization>\n\t\t<colorings>\n\t\t\t<class rgba="232,139,15,100" visible="false">wunderZone</class>\n\t\t</colorings>\n\t</visualization>\n\t<reporting>\n\t\t<reportElements />\n\t</reporting>\n</scenario>'

    project_xml_text = xml_header + scenario_header + floor_line + scenario_bottom + origin_head + origin_bottom + destination_lines + filler + floorProps + filler2 + bottom

    with open(project_filename, 'w') as f:
        f.write(project_xml_text)
    f.close()

def export2crowdit_project(preprocessed_lines, origins, destinations, crowdit_descr):

    root_folder = 'C:\\Users\\ga78jem\\Documents\\Crowdit'
    crowdit_folder = os.path.join(root_folder, crowdit_descr)
    if not os.path.isdir(crowdit_folder): os.mkdir(crowdit_folder)

    header = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    floor_header = '<floor isoDate="2022-05-24T13:17:59.807407200Z" srcFile="INSERT_SRC_DXF" xmlFormat="0.10.2">\n'
    layer_0 = '\t<layer id="0"/>\n'
    layer_A_WALL = '\t<layer id="A-WALL-____-MCUT">\n'
    layer_A_WALL_bottom = '\t</layer>\n'
    layer_crowdit = '\t<layer id="crowdit">\n'
    layer_crowdit_bottom = '\t</layer>\n'
    floor_bottom = '</floor>'
    wall_content = ''
    crowdit_content = ''

    for idx, line in enumerate(preprocessed_lines):
        wall = f'\t\t<wall closed="false" id="w{idx+2}">\n'
        for point in line:
            x = {round(point[0], 2)}
            y = {round(point[1], 2)}
            wall += f'\t\t\t<point x="{round(point[0], 2)}" y="{round(point[1], 2)}"/>\n'
        wall += '\t\t</wall>\n'
        wall_content += wall

    for id_o, origin in enumerate(origins):
        destination = destinations[id_o]
        s1x = origin[0][0]
        s1y = origin[0][1]
        s2x = origin[1][0]
        s2y = origin[1][1]

        d1x = destination[0][0]
        d1y = destination[0][1]
        d2x = destination[1][0]
        d2y = destination[1][1]
        
        crowdit_content = f'\t\t<wunderZone id="simObj-192">\n'
        crowdit_content += f'\t\t\t<point x="{s1x}" y="{s1y}"/>\n'
        crowdit_content += f'\t\t\t<point x="{s2x}" y="{s1y}"/>\n'
        crowdit_content += f'\t\t\t<point x="{s2x}" y="{s2y}"/>\n'
        crowdit_content += f'\t\t\t<point x="{s1x}" y="{s2y}"/>\n'
        crowdit_content += '\t\t</wunderZone>\n'

        crowdit_content += f'\t\t<wunderZone id="simObj-198">\n'
        crowdit_content += f'\t\t\t<point x="{d1x}" y="{d1y}"/>\n'
        crowdit_content += f'\t\t\t<point x="{d2x}" y="{d1y}"/>\n'
        crowdit_content += f'\t\t\t<point x="{d2x}" y="{d2y}"/>\n'
        crowdit_content += f'\t\t\t<point x="{d1x}" y="{d2y}"/>\n'
        crowdit_content += '\t\t</wunderZone>\n'

        xml_text = header + floor_header + layer_0 + layer_A_WALL + wall_content + layer_A_WALL_bottom + layer_crowdit + crowdit_content + layer_crowdit_bottom + floor_bottom
        
        variation_folder = os.path.join(crowdit_folder, f'variation_{id_o}')
        if not os.path.isdir(variation_folder): os.mkdir(variation_folder)

        project_folder = os.path.join(variation_folder, f'project_{crowdit_descr.split("__")[0]}_{id_o}_res')
        if not os.path.isdir(project_folder): os.mkdir(project_folder)

        geometry_folder = os.path.join(project_folder, 'geometry')
        if not os.path.isdir(geometry_folder): os.mkdir(geometry_folder)

        meta_folder = os.path.join(project_folder, 'meta')
        if not os.path.isdir(meta_folder): os.mkdir(meta_folder)

        write_crowdit_meta_folder(meta_folder)

        floorname = os.path.join(geometry_folder, f'{crowdit_descr.split("__")[-1]}_variation_{id_o}.floor')

        write_crowdit_project_file(
            project_filename = os.path.join(variation_folder, f'project_{crowdit_descr.split("__")[0]}_{id_o}.crowdit'), 
            project_name = f'project_{crowdit_descr.split("__")[0]}_{id_o}',
            floor_filename = f'geometry/{crowdit_descr.split("__")[-1]}_variation_{id_o}.floor'
            )

        with open(floorname, 'w') as f:
            f.write(xml_text)
        f.close()