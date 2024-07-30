from trainers.bert_classifier import train_hf
from data_loading.models_dataset import ModelDataset 

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='modelset', choices=['modelset', 'ecore', 'mar'])
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--reload', action='store_true')
    return parser.parse_args()


def run(args):
    reload = args.reload
    ecore = ModelDataset('ecore', reload=reload)
    modelset = ModelDataset('modelset', reload=reload, remove_duplicates=True)
    # mar = ModelDataset('mar-ecore-github', reload=reload)


    datasets = {
        'ecore': ecore,
        'modelset': modelset,
        # 'mar': mar
    }
    bert = 'bert-base-uncased'
    longformer = 'allenai/longformer-base-4096'
    models = [bert, longformer]
    for name, dataset in datasets.items():
        for model in models:
            print(f'Training {name} using {model}')
            train_hf(bert, dataset, epochs= 3 if dataset == 'ecore' else 10)
            print(f'Finished training {name} using {model}')
            print('-----------------------------------')