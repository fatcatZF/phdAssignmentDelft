from lxml import etree
import os
import sys
import ast
import pandas as pd

n_file = "files_e3/onramp_network.net.xml"
in_file_ab = "files_e3/gen_ab.xml"
in_file_ob = "files_e3/gen_ob.xml"
add = "files_e3/output_req.xml,files_e3/add_loop.xml"
output = "files_e3/output/edge_out.xml"
fcd_output = "files_e3/output/fcd_out.xml"

def plot_demand(lane):
    if lane == 'ab':
        in_plot = "files_e3/gen_ab.xml"
        tit = 'A -> B'
    elif lane == 'ob':
        in_plot = "files_e3/gen_ob.xml"
        tit = 'O -> B'
    else:
        raise ValueError("Please specify either 'ab' or 'ob' as input.")
    dataTree = etree.parse(in_plot)
    dic_flow = {}
    root = dataTree.getroot()
    for e in root.getchildren():
        if 'vehsPerHour' in e.attrib.keys():
            id_at = e.attrib['id']
            dic_flow[id_at] = {}
            dic_flow[id_at]['begin'] = e.attrib['begin']
            dic_flow[id_at]['end'] = e.attrib['end']
            dic_flow[id_at]['vehsPerHour'] = e.attrib['vehsPerHour']

    df_flow = pd.DataFrame.from_dict(dic_flow, orient='index')
    df_flow['begin'] = df_flow['begin'].astype(float)
    df_flow['end'] = df_flow['end'].astype(float)
    df_flow['vehsPerHour'] = df_flow['vehsPerHour'].astype(int)
    df_flow.plot.bar(x='begin', y='vehsPerHour', legend=False, title=f'Flow {tit} per Time Step')

def call_sim_base(gui=False):
    if gui:
        sumoCMD = "sumo-gui -n files_e3/onramp_network.net.xml -r files_e3/gen_ab.xml,files_e3/gen_ob.xml -a files_e3/output_req.xml,files_e3/add_loop.xml --fcd-output files_e3/output/fcd_out.xml --lateral-resolution 2.5"
    else:
        sumoCMD = "sumo -n files_e3/onramp_network.net.xml -r files_e3/gen_ab.xml,files_e3/gen_ob.xml -a files_e3/output_req.xml,files_e3/add_loop.xml --fcd-output files_e3/output/fcd_out.xml --lateral-resolution 2.5"
    os.system(sumoCMD)

def output_to_df(filename):
    dataTree = etree.parse(filename)
    root = dataTree.getroot()
    it = 0
    val_ls = []
    int_ls = ['id', 'sampledSeconds', 'traveltime', 'overlapTraveltime', 'density', 'laneDensity', 'occupancy','waitingTime','timeLoss',
             'speed','speedRelative','departed','arrived','entered','left','laneChangedFrom', 'laneChangedTo', 'end', 'begin']
    for e in root.getchildren():
        for t in e.getchildren():
            att_v = t.attrib
            att_v['begin'] = e.attrib['begin']
            att_v['end'] = e.attrib['end']
            att_v = ast.literal_eval(f"{att_v}")
            val_ls.append(att_v)
    df = pd.DataFrame(val_ls)

    for col in int_ls:
        if col != 'id':
            df[col] = df[col].astype(float)
            
    return df

def veh_output_to_df(filename):
    dataTree = etree.parse(filename)
    root = dataTree.getroot()
    it = 0
    val_ls = []
    for e in root.getchildren():
        for t in e.getchildren():
            att_v = t.attrib
            att_v['time'] = e.attrib['time']
            att_v = ast.literal_eval(f"{att_v}")
            val_ls.append(att_v)
    df = pd.DataFrame(val_ls)      
    return df

def read_output():
    df = output_to_df(output)
    df_veh = veh_output_to_df(fcd_output)
    return df, df_veh

def run_base(gui=False):
    call_sim_base(gui)
    return read_output()
