"""
GLAM4CM Main Entry Point

This module serves as the main entry point for the GLAM4CM framework, providing
a unified interface to all downstream tasks including BERT-based models, GNNs,
and CM-GPT models for conceptual modeling tasks.

The framework supports multiple task types:
- Dataset creation and preprocessing
- Node, edge, and graph classification
- Link prediction
- Text classification
- Causal modeling with CM-GPT

Author: Syed Juned Ali
Email: syed.juned.ali@tuwien.ac.at
"""

import argparse
from glam4cm.downstream_tasks import (
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

from glam4cm.downstream_tasks import cm_gpt_pretraining
from glam4cm.downstream_tasks import cm_gpt_node_classification
from glam4cm.downstream_tasks import cm_gpt_edge_classification
from glam4cm.downstream_tasks.bert_graph_classification_comp import (
    get_parser as bert_comp_parse_args,
)
from glam4cm.downstream_tasks.bert_graph_classification import (
    get_parser as bert_parse_args,
)
from glam4cm.downstream_tasks.gnn_graph_cls import get_parser as gnn_parse_args
from glam4cm.downstream_tasks.create_dataset import (
    get_parser as create_dataset_parse_args,
)
from glam4cm.downstream_tasks.bert_link_prediction import (
    get_parser as bert_lp_parse_args,
)
from glam4cm.downstream_tasks.gnn_edge_classification import (
    get_parser as gnn_ec_parse_args,
)
from glam4cm.downstream_tasks.gnn_link_prediction import get_parser as gnn_lp_parse_args
from glam4cm.downstream_tasks.bert_edge_classification import (
    get_parser as bert_ec_parse_args,
)
from glam4cm.downstream_tasks.gnn_node_classification import (
    get_parser as gnn_nc_parse_args,
)
from glam4cm.downstream_tasks.bert_node_classification import (
    get_parser as bert_nc_parse_args,
)
from glam4cm.downstream_tasks.cm_gpt_pretraining import get_parser as cm_gpt_parse_args
from glam4cm.downstream_tasks.cm_gpt_node_classification import (
    get_parser as cm_gpt_nc_parse_args,
)
from glam4cm.downstream_tasks.cm_gpt_edge_classification import (
    get_parser as cm_gpt_ec_parse_args,
)


# Task ID to task name mapping for user-friendly identification
tasks = {
    0: "Create Dataset",
    1: "BERT Graph Classification Comparison",
    2: "BERT Graph Classification",
    3: "BERT Node Classification",
    4: "BERT Link Prediction",
    5: "BERT Edge Classification",
    6: "GNN Graph Classification",
    7: "GNN Node Classification",
    8: "GNN Link Prediction",
    9: "GNN Edge Classification",
    10: "CM-GPT Causal Modeling",
    11: "CM-GPT Node Classification",
    12: "CM-GPT Edge Classification",
}


# Task ID to handler function and argument parser mapping
# Each task has a corresponding run function and argument parser
tasks_handler_map = {
    0: (create_dataset.run, create_dataset_parse_args),
    1: (bert_graph_classification_comp.run, bert_comp_parse_args),
    2: (bert_graph_classification.run, bert_parse_args),
    3: (bert_node_classification.run, bert_nc_parse_args),
    4: (bert_link_prediction.run, bert_lp_parse_args),
    5: (bert_edge_classification.run, bert_ec_parse_args),
    6: (gnn_graph_cls.run, gnn_parse_args),
    7: (gnn_node_classification.run, gnn_nc_parse_args),
    8: (gnn_link_prediction.run, gnn_lp_parse_args),
    9: (gnn_edge_classification.run, gnn_ec_parse_args),
    10: (cm_gpt_pretraining.run, cm_gpt_parse_args),
    11: (cm_gpt_node_classification.run, cm_gpt_nc_parse_args),
    12: (cm_gpt_edge_classification.run, cm_gpt_ec_parse_args),
}


def main():
    """
    Main entry point for the GLAM4CM framework.

    This function:
    1. Parses command-line arguments to determine the task to run
    2. Routes to the appropriate task handler with parsed arguments
    3. Provides help information for specific tasks when requested

    Command-line arguments:
        --task_id: ID of the task to run (0-12)
        --th/--task_help: Show help for the specified task

    Example usage:
        python -m glam4cm.run --task_id 3 --help  # Show help for BERT Node Classification
        python -m glam4cm.run --task_id 6 --model_name GCNConv  # Run GNN Graph Classification
    """
    # Create the main argument parser
    main_parser = argparse.ArgumentParser(
        description="Train ML models on conceptual models using GLAM4CM framework"
    )

    # Add task selection argument
    main_parser.add_argument(
        "--task_id",
        type=int,
        required=True,
        help=f"ID of the task to run. Available tasks:\n{'\n'.join(f'{k}: {v}' for k, v in tasks.items())}",
        choices=list(tasks.keys()),
        default=0,
    )

    # Add task-specific help argument
    main_parser.add_argument(
        "--th",
        "--task_help",
        action="store_true",
        help="Show help for the task specified by --task_id",
    )

    # Parse known arguments to handle both main and task-specific arguments
    args, remaining_args = main_parser.parse_known_args()

    # Check if any arguments were provided
    if not any(vars(args).values()):
        print("No arguments provided. Please provide arguments to run the task.")
        main_parser.print_help()
        exit(1)

    # Handle task-specific help requests
    if any(x in remaining_args for x in ["-th", "--task_help"]):
        task_id = args.task_id
        task_handler, task_parser = tasks_handler_map[task_id]
        print("Help for task:", tasks[task_id])
        task_parser().print_help()
        exit(0)

    # Log the task being executed
    print("Running GLAM4CM with:", vars(args))

    # Execute the selected task
    task_id = args.task_id
    task_handler, task_parser = tasks_handler_map[task_id]

    # Parse task-specific arguments and run the task
    task_args = task_parser().parse_args(remaining_args)
    task_handler(task_args)


if __name__ == "__main__":
    main()
