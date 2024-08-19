from trainers.bert_classifier import train_hf
from data_loading.models_dataset import EcoreModelDataset 

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ecore_555', choices=['modelset', 'ecore_555', 'mar-ecore-github'])
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--reload', action='store_true')
    return parser.parse_args()


def run(args):

    config_params = dict(
        timeout = args.timeout,
        min_enr = args.min_enr,
        min_edges = args.min_edges,
        remove_duplicates = args.remove_duplicates,
        reload=args.reload
    )
    dataset_name = args.dataset
    dataset = EcoreModelDataset(dataset_name, **config_params)

    model_name = args.model_name
    epochs = args.epochs
    train_hf(model_name, dataset, epochs=epochs)