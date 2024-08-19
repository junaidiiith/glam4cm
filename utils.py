import argparse
from collections import Counter
import random
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import os
import fnmatch
import json
from typing import List
import xmltodict


def find_files_with_extension(root_dir, extension):
    matching_files: List[str] = list()

    # Recursively search for files with the specified extension
    for root, _, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, f'*.{extension}'):
            matching_files.append(os.path.join(root, filename))

    return matching_files


def xml_to_json(xml_string):
    xml_dict = xmltodict.parse(xml_string)
    json_data = json.dumps(xml_dict, indent=4)
    return json_data


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def randomize_features(d, num_feats):
    for i, data in enumerate(d):
        num_nodes = data.num_nodes
        num_edges = data.overall_edge_index.shape[1] if hasattr(data, 'overall_edge_index') else data.edge_index.shape[1]
        d[i].x = torch.randn((num_nodes, num_feats))
        d[i].edge_attr = torch.randn((num_edges, num_feats))
    return d


def merge_namespaces(ns1, ns2):
    merged_ns = argparse.Namespace()
    merged_ns.__dict__.update(ns1.__dict__)
    merged_ns.__dict__.update(ns2.__dict__)
    return merged_ns
