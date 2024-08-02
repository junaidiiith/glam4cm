import os
import torch


BERT_MODEL = 'bert-base-uncased'
FAST_TEXT_MODEL = 'uml-fasttext.bin'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

seed = 42
datasets_dir = 'datasets'
ecore_json_path = os.path.join(datasets_dir, 'ecore_555/ecore_555.jsonl')
mar_json_path = os.path.join(datasets_dir, 'mar-ecore-github/ecore-github.jsonl')
modelsets_uml_json_path = os.path.join(datasets_dir, 'modelset/uml.jsonl')
modelsets_ecore_json_path = os.path.join(datasets_dir, 'modelset/ecore.jsonl')


graph_data_dir = 'datasets/graph_data'

# Path: settings.py
