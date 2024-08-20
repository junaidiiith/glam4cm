from test import (
    bert_graph_classification,
    bert_link_prediction,
    bert_node_classification,
    gnn_edge_classification,
    gnn_link_prediction,
    gnn_graph_cls,
    create_dataset,
    bert_edge_classification,
    gnn_node_classification
)
from test.bert_graph_classification import get_parser as bert_parse_args
from test.gnn_graph_cls import get_parser as gnn_parse_args
from test.create_dataset import get_parser as create_dataset_parse_args
from test.bert_link_prediction import get_parser as bert_lp_parse_args
from test.gnn_edge_classification import get_parser as gnn_ec_parse_args
from test.gnn_link_prediction import get_parser as gnn_lp_parse_args
from test.bert_edge_classification import get_parser as bert_ec_parse_args
from test.gnn_node_classification import get_parser as gnn_nc_parse_args
from test.bert_node_classification import get_parser as bert_nc_parse_args


tasks_handler_map = {
    1: (create_dataset.run, create_dataset_parse_args),
    2: (bert_graph_classification.run, bert_parse_args),
    3: (bert_edge_classification.run, bert_ec_parse_args),
    4: (bert_link_prediction.run, bert_lp_parse_args),
    5: (bert_node_classification.run, bert_nc_parse_args),
    6: (gnn_graph_cls.run, gnn_parse_args),
    7: (gnn_edge_classification.run, gnn_ec_parse_args),
    8: (gnn_link_prediction.run, gnn_lp_parse_args),
    9: (gnn_node_classification.run, gnn_nc_parse_args)
}


if __name__ == '__main__':
    task_id = 7
    hander, task_args = tasks_handler_map[task_id]
    hander(task_args())
