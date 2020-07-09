import json
import os
def save_single_dict2json(s_dict, json_file):
    with open(json_file, 'w') as of_:
        s_dict = json.dumps(s_dict)
        json.dump(s_dict, json_file)
        of_.write(s_dict)


def load_json_list(src_file):
    # src_file = '/home/zwei/Dev/Adobe2018/stockimage/Emotion_6_0100.json'
    video_list = []
    with open(src_file, 'r') as outf:
        for line in outf:
            s_item = json.loads(line)
            video_list.append(s_item)
    return video_list


def save_json_list(data_list, save_file):
    with open(save_file, 'w') as of_:
        for s_line in data_list:
            of_.write(json.dumps(s_line)+'\n')

def load_json(json_file):
    with open(json_file) as j_reader:
        data = json.load(j_reader)
    return data


def load_json2dict(json_file):
    assert os.path.isfile(json_file), "cannot find {}".format(json_file)
    with open(json_file, 'r') as of_:
        data = json.load(of_)
    assert isinstance(data, dict), type(data)
    return data