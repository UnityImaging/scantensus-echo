import json
import logging

from typing import List, Dict
from pathlib import Path

import numpy as np

from Scantensus.utils.heatmaps import interpolate_curve

def add_snake_to_curve_dict(curves_dict, snake, curve_name, type):

    if type == 'curve':
        out_type = 'curves'
        out_type_node = 'nodes'
    elif type == 'point':
        out_type = 'nodes'
        out_type_node = 'node'
    else:
        raise Exception

    curves_dict[out_type][curve_name] = {}
    curves_dict[out_type][curve_name]['thready_prediction'] = {}
    curves_dict[out_type][curve_name]['thready_prediction']['format'] = 2
    curves_dict[out_type][curve_name]['thready_prediction']['n'] = 1
    curves_dict[out_type][curve_name]['thready_prediction']['vis'] = 'seen'


    if type == 'curve':
        node_keys = []
        for i, node in enumerate(snake):
            temp = {}
            temp['i2'] = 0.01
            temp['x'] = int(node[1])
            temp['y'] = int(node[0])
            temp['z'] = float(node[2])
            node_keys.append(temp)

        curves_dict[out_type][curve_name]['thready_prediction']['instances'] = [{out_type_node: node_keys}]

    if type == 'point':
        node_key = {}
        node_key['i2'] = 0.01
        node_key['x'] = int(snake[0, 1])
        node_key['y'] = int(snake[0, 0])
        node_key['z'] = float(snake[0, 2])

        curves_dict[out_type][curve_name]['thready_prediction']['instances'] = [node_key]

    return curves_dict


def add_snake_to_curve_dict_csv(curves_dict, snake, curve_name):
    curves_dict[curve_name + "_y"] = " ".join([str(value) for value in snake[..., 0]])
    curves_dict[curve_name + "_x"] = " ".join([str(value) for value in snake[..., 1]])

    return curves_dict

def get_keypoint_names_and_colors_from_json(json_path):
    with open(json_path, "r") as read_file:
        data = json.load(read_file)

    keypoint_names = list(data.keys())
    keypoint_cols = []

    for keypoint in keypoint_names:
        r, g, b = data[keypoint]['rgb'].split()
        keypoint_cols.append([float(r), float(g), float(b)])

    return keypoint_names, keypoint_cols


def convert_labels_to_csv(out_labels_dict: dict, labels_to_process: List[str], project: str):

    out_csv_list = []

    for unity_f_code, data in out_labels_dict.items():
        for label in labels_to_process:

            curve_y_str = data['labels'][label]['y']
            curve_y = curve_str_to_list(curve_y_str)

            curve_x_str = data['labels'][label]['x']
            curve_x = curve_str_to_list(curve_x_str)

            if len(curve_x) >= 3:
                curve = np.stack((curve_y, curve_x))
                curve = interpolate_curve(curve.T)
                curve_len = np.sum(np.sqrt(np.sum((curve[1:, 0:2] - curve[:-1, 0:2]) ** 2, axis=-1)), axis=-1)
                curve_len = str(round(float(curve_len), 2))
            else:
                curve_len = ''

            out_csv = {"file": unity_f_code + ".png",
                       "label": label,
                       "user": "scantensus-echo",
                       "time": "",
                       "project": project,
                       "type": data['labels'][label]['type'],
                       "value_y": curve_y_str,
                       "value_x": curve_x_str,
                       "conf": data['labels'][label]['conf'],
                       "curve_len": curve_len}

            out_csv_list.append(out_csv)

    return out_csv_list



def get_cols_from_firebase_project_config(config_data: Dict):

    ## pass it this firebase_data['fiducial'][project]['config']

    col_data = config_data

    col_dict = {}

    try:
        for item in col_data['nodes'].items():
            item_name = item[0]
            col = item[1]['color'][1:]
            r = int(col[:2], 16) / 255
            g = int(col[2:4], 16) / 255
            b = int(col[4:6], 16) / 255
            col_dict[item_name] = [r, g, b]
    except Exception as e:
        logging.info(f'get_cols_from_firebase_project_config: no nodes')

    try:
        for item in col_data['curves'].items():
            item_name = item[0]
            col = item[1]['color'][1:]
            r = int(col[:2], 16) / 255
            g = int(col[2:4], 16) / 255
            b = int(col[4:6], 16) / 255
            col_dict[item_name] = [r, g, b]
    except Exception as e:
        logging.info(f'get_cols_from_firebase_project_config: no curves')

    return col_dict




def convert_labels_to_firebase(out_labels_dict: Dict,
                               labels_to_process: List[str],
                               firebase_reverse_dict: Dict,
                               user_name: str = "thready_prediction"):

    out_firebase_dict = {}

    for unity_code, unity_data in out_labels_dict.items():

        data = unity_data['labels']

        out_unity_code = Path(unity_code).stem + ":png"

        out_firebase_dict[out_unity_code] = {}
        out_firebase_dict[out_unity_code]['curves'] = {}
        out_firebase_dict[out_unity_code]['nodes'] = {}

        for label_name in labels_to_process:
            firebase_label_name = firebase_reverse_dict[label_name]
            path_x = data[label_name]['x']
            path_y = data[label_name]['y']
            path_conf = data[label_name]['conf']
            path_type = data[label_name]['type']

            if path_type == 'curve':
                firebase_type = 'curves'
                firebase_type_node = 'nodes'
            elif path_type == 'point':
                firebase_type = 'nodes'
                firebase_type_node = 'node'
            else: #blurred or off
                continue

            out_firebase_dict[out_unity_code][firebase_type][firebase_label_name] = {}

            working_dict = out_firebase_dict[out_unity_code][firebase_type][firebase_label_name]

            working_dict[user_name] = {}
            working_dict[user_name]['format'] = 2
            working_dict[user_name]['n'] = 1
            working_dict[user_name]['vis'] = 'seen'

            if path_type == 'curve':

                node_keys = []
                xs = path_x.split(" ")
                ys = path_y.split(" ")
                confs = path_conf.split(" ")
                num_nodes = len(xs)

                for i, (x, y, conf) in enumerate(zip(xs, ys, confs)):
                    temp = {}
                    temp['i2'] = 0.01
                    temp['x'] = int(float(x))
                    temp['y'] = int(float(y))
                    temp['z'] = float(conf)
                    if i == 0 and label_name == "curve-lv-endo":
                        temp['nodeKey'] = 'mv-ant-hinge'
                    if i == (num_nodes // 2) and label_name == "curve-lv-endo":
                        temp['nodeKey'] = 'lv-apex-endo'
                    if i == (num_nodes-1) and label_name == "curve-lv-endo":
                        temp['nodeKey'] = 'mv-post-hinge'
                    node_keys.append(temp)

                working_dict[user_name]['instances'] = [{firebase_type_node: node_keys}]

            if path_type == 'point':
                node_key = {}
                node_key['i2'] = 0.01
                node_key['x'] = int(float(path_x))
                node_key['y'] = int(float(path_y))
                node_key['z'] = float(path_conf)
                node_key['nodeKey'] = label_name

                working_dict[user_name]['instances'] = [node_key]

    return out_firebase_dict


def curve_list_to_str(curve_list: List, round_digits=1):
    out = " ".join([str(round(value, round_digits)) for value in curve_list])
    return out


def curve_str_to_list(curve_str: str):
    out = [(float(value)) for value in curve_str.split()]
    return out


def get_labels_from_firebase_project_data(project_data, project, mapping_data):

    logging.info(f'Project: {project}')
    # project_data = firebase_data['fiducial'][project]['labels']

    db = []

    for image_name, image_data in project_data.items():

        if image_name.startswith('01-') or image_name.startswith('02-'):
            file = image_name.replace("_(1)", "").replace("_(2)", "").replace("_(3)", "")
            file = file.replace(":", ".")
        else:
            file_name_mangled_split = image_name.split('~')
            project_id = file_name_mangled_split[0]
            naming_scheme = file_name_mangled_split[1]
            file = file_name_mangled_split[2].replace(":", ".")
            file = file.replace("_(1)", "").replace("_(2)", "").replace("_(3)", "")

            if naming_scheme != "unique":
                continue

            if project_id != project:
                logging.warning(f"project_id / project mismatch {project_id} / {project}")
                continue

        user_dict = {}

        for user_item in image_data['events'].items():
            user_code = user_item[0]
            user = user_item[1]['user']
            time_stamp = user_item[1]['t']

            user_data = {}
            user_data['user'] = user
            user_data['time_stamp'] = time_stamp

            user_dict[user_code] = user_data

        nodes = image_data.get('nodes', {})
        curves = image_data.get('curves', {})

        labels = {**nodes, **curves}

        for label_name_old, label_data_all in labels.items():

            label_name = mapping_data.get(label_name_old, None)

            if label_name is None:
                logging.warning(f"Missing mapping for {label_name_old}")
                continue

            for user_code, node_data in label_data_all.items():
                true_user = user_dict[user_code]['user']
                true_time_stamp = user_dict[user_code]['time_stamp']

                if 'format' in node_data:
                    format = node_data['format']
                    vis = node_data.get('vis', None)
                    try:
                        node_instance = node_data['instances'][0]
                        if 'node' in node_instance:
                            node_list = [node_instance['node']]
                        elif 'nodes' in node_instance:
                            node_list = node_instance['nodes']
                    except Exception as e:
                        logging.warning(f"no node list. vis {vis}")
                        vis = 'blurred'
                else:
                    format = 1
                    vis = 'seen'
                    if type(node_data) is list:
                        node_list = node_data
                    else:
                        node_list = [node_data]

                if vis == "blurred":
                    out = {'project': project,
                           'file': file,
                           'user': true_user,
                           'time': true_time_stamp,
                           'label': label_name,
                           'vis': 'blurred',
                           'value_x': '',
                           'value_y': '',
                           'curve_len': ''
                           }

                    db.append(out)
                    continue

                if vis == 'off':
                    out = {'project': project,
                           'file': file,
                           'user': true_user,
                           'time': true_time_stamp,
                           'label': label_name,
                           'vis': 'off',
                           'value_x': '',
                           'value_y': '',
                           'curve_len': ''
                           }
                    db.append(out)
                    continue

                curve_x = []
                curve_y = []

                for node in node_list:
                    curve_x.append(node['x'])
                    curve_y.append(node['y'])

                if len(curve_x) >= 3:
                    curve = np.stack((curve_y, curve_x))
                    curve = interpolate_curve(curve.T)
                    curve_len = np.sum(np.sqrt(np.sum((curve[1:, 0:2] - curve[:-1, 0:2]) ** 2, axis=-1)), axis=-1)
                    curve_len = str(round(float(curve_len), 2))
                else:
                    curve_len = ''

                out = {'project': project,
                       'file': file,
                       'user': true_user,
                       'time': true_time_stamp,
                       'label': label_name,
                       'vis': 'seen',
                       'value_x': curve_list_to_str(curve_x),
                       'value_y': curve_list_to_str(curve_y),
                       'curve_len': curve_len
                       }

                db.append(out)

    return db

