from test import (
    bert_train,
    gnn_graph_cls,
    create_dataset,
    bert_link_predictor,
    bert_edge_classification,
    gnn_link_prediction
)
from test.bert_train import parse_args as bert_parse_args
from test.gnn_graph_cls import parse_args as gnn_parse_args
from test.create_dataset import parse_args as create_dataset_parse_args
from test.bert_link_predictor import parse_args as bert_lp_parse_args
from test.gnn_link_prediction import parse_args as gnn_lp_parse_args
from test.bert_edge_classification import parse_args as bert_ec_parse_args


if __name__ == '__main__':

    test_task = 4
    if test_task == 0:
        create_dataset.run(create_dataset_parse_args())
    elif test_task == 1:
        bert_train.run(bert_parse_args())
    elif test_task == 2:
        gnn_graph_cls.run(gnn_parse_args())
    elif test_task == 3:
        bert_link_predictor.run(bert_lp_parse_args())
    elif test_task == 4:
        bert_edge_classification.run(bert_ec_parse_args())
    elif test_task == 5:
        gnn_link_prediction.run(gnn_lp_parse_args())
    else:
        raise ValueError(f'Invalid test task: {test_task}')
