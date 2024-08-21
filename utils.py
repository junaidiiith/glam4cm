from argparse import ArgumentParser
import random
import numpy as np
import torch
import os
import fnmatch
import json
from typing import List
import xmltodict
from torch_geometric.data import Data


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


def randomize_features(dataset: List[Data], num_feats):
    for data in dataset:
        num_nodes = data.num_nodes
        num_edges = data.overall_edge_index.shape[1] if hasattr(data, 'overall_edge_index') else data.edge_index.shape[1]
        data.x = torch.randn((num_nodes, num_feats))
        data.edge_attr = torch.randn((num_edges, num_feats))


def merge_argument_parsers(p1: ArgumentParser, p2: ArgumentParser):
    merged_parser = ArgumentParser(description="Merged Parser")

    # Combine arguments from parser1
    for action in p1._actions:
        if action.dest != "help":  # Skip the help action
            merged_parser._add_action(action)

    # Combine arguments from parser2
    for action in p2._actions:
        if action.dest != "help":  # Skip the help action
            merged_parser._add_action(action)

    return merged_parser


def is_meaningful_line(line: str):
    stripped_line: str = line.strip()
    # Ignore empty lines, comments, and docstrings
    if stripped_line == "" or stripped_line.startswith("#") or stripped_line.startswith('"""') or stripped_line.startswith("'''"):
        return False
    return True

def count_lines_of_code_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        meaningful_lines = [line for line in lines if is_meaningful_line(line)]
    return len(meaningful_lines)

def count_total_lines_of_code(directory):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                total_lines += count_lines_of_code_in_file(file_path)
    return total_lines