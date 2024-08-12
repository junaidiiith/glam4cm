from test import bert_train
from test import gnn_graph_cls
from test import create_dataset
from test import bert_link_predictor
from test.bert_train import parse_args as bert_parse_args
from test.gnn_graph_cls import parse_args as gnn_parse_args
from test.create_dataset import parse_args as create_dataset_parse_args
from test.bert_link_predictor import parse_args as bert_lp_parse_args


test_tasks = {
    0: 'create_dataset',
    1: 'bert_train',
    2: 'gnn_graph_cls',
    3: 'bert_lp'
}

if __name__ == '__main__':

    test_task = 3
    if test_task == 0:
        create_dataset.run(create_dataset_parse_args())
    elif test_task == 1:
        bert_train.run(bert_parse_args())
    elif test_task == 2:
        gnn_graph_cls.run(gnn_parse_args())
    elif test_task == 3:
        bert_link_predictor.run(bert_lp_parse_args())
    else:
        raise ValueError(f'Invalid test task: {test_task}')
