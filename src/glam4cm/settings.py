import os
import torch

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



BERT_MODEL = 'bert-base-uncased'
WORD2VEC_MODEL = 'word2vec'
TFIDF_MODEL = 'tfidf'
FAST_TEXT_MODEL = 'uml-fasttext.bin'

W2V_CONFIG = dict(
    epoch=100,
    dim=128,
    ws=5,
    minCount=1,
    thread=4,
    model='skipgram'
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_float32_matmul_precision('high')


seed = 42
datasets_dir = 'datasets'
ecore_json_path = os.path.join(datasets_dir, 'ecore_555/ecore_555.jsonl')
mar_json_path = os.path.join(datasets_dir, 'mar-ecore-github/ecore-github.jsonl')
modelsets_uml_json_path = os.path.join(datasets_dir, 'modelset/uml.jsonl')
modelsets_ecore_json_path = os.path.join(datasets_dir, 'modelset/ecore.jsonl')


graph_data_dir = 'datasets/graph_data'

# Path: settings.py


LP_TASK_EDGE_CLS = 'edge_cls'
LP_TASK_LINK_PRED = 'lp'


EPOCH = 'epoch'
LOSS = 'loss'
TRAIN_LOSS = 'train_loss'
TEST_LOSS = 'test_loss'
TEST_ACC = 'test_acc'

TRAINING_PHASE = 'train'
VALIDATION_PHASE = 'val'
TESTING_PHASE = 'test'
