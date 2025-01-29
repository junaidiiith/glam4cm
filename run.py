import argparse
from downstream_tasks import (
    bert_graph_classification_comp,
    bert_graph_classification,
    bert_node_classification,
    bert_edge_classification,
    bert_link_prediction,

    gnn_graph_cls,
    gnn_node_classification,
    gnn_edge_classification,
    gnn_link_prediction,
    create_dataset,
)

from downstream_tasks import cm_gpt_pretraining
from downstream_tasks import cm_gpt_node_classification
from downstream_tasks import cm_gpt_edge_classification
from downstream_tasks.bert_graph_classification_comp import get_parser as bert_comp_parse_args
from downstream_tasks.bert_graph_classification import get_parser as bert_parse_args
from downstream_tasks.gnn_graph_cls import get_parser as gnn_parse_args
from downstream_tasks.create_dataset import get_parser as create_dataset_parse_args
from downstream_tasks.bert_link_prediction import get_parser as bert_lp_parse_args
from downstream_tasks.gnn_edge_classification import get_parser as gnn_ec_parse_args
from downstream_tasks.gnn_link_prediction import get_parser as gnn_lp_parse_args
from downstream_tasks.bert_edge_classification import get_parser as bert_ec_parse_args
from downstream_tasks.gnn_node_classification import get_parser as gnn_nc_parse_args
from downstream_tasks.bert_node_classification import get_parser as bert_nc_parse_args
from downstream_tasks.cm_gpt_pretraining import get_parser as cm_gpt_parse_args
from downstream_tasks.cm_gpt_node_classification import get_parser as cm_gpt_nc_parse_args
from downstream_tasks.cm_gpt_edge_classification import get_parser as cm_gpt_ec_parse_args


tasks = {
    0: 'Create Dataset',

    1: 'BERT Graph Classification Comparison',
    2: 'BERT Graph Classification',
	3: 'BERT Node Classification',
    4: 'BERT Link Prediction',
    5: 'BERT Edge Classification',
    
    
    6: 'GNN Graph Classification',
    7: 'GNN Node Classification',
    8: 'GNN Edge Classification',
    9: 'GNN Link Prediction',
    10: 'CM-GPT Causal Modeling',
    11: 'CM-GPT Node Classification',
    12: 'CM-GPT Edge Classification'
}


tasks_handler_map = {
    0: (create_dataset.run, create_dataset_parse_args),
    1: (bert_graph_classification_comp.run, bert_comp_parse_args),
    2: (bert_graph_classification.run, bert_parse_args),
    3: (bert_node_classification.run, bert_nc_parse_args),
    4: (bert_link_prediction.run, bert_lp_parse_args),
    5: (bert_edge_classification.run, bert_ec_parse_args),
    6: (gnn_graph_cls.run, gnn_parse_args),
    7: (gnn_node_classification.run, gnn_nc_parse_args),
    8: (gnn_edge_classification.run, gnn_ec_parse_args),
    9: (gnn_link_prediction.run, gnn_lp_parse_args),
    10: (cm_gpt_pretraining.run, cm_gpt_parse_args),
    11: (cm_gpt_node_classification.run, cm_gpt_nc_parse_args),
    12: (cm_gpt_edge_classification.run, cm_gpt_ec_parse_args)
}


if __name__ == '__main__':

    main_parser = argparse.ArgumentParser(description="Main parser")
    main_parser.add_argument('--task_id', type=int, required=True, help=f'ID of the task to run. Options are: {tasks}', choices=tasks.keys(), default=0)
    args, remaining_args = main_parser.parse_known_args()

    task_id = args.task_id
    hander, task_parser = tasks_handler_map[task_id]
    task_args = task_parser().parse_args(remaining_args)
    hander(task_args)