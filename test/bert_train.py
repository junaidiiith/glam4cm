from data_loading.dataset import Dataset
from trainers.bert_classifier import train_hf


def run():
    reload = False
    ecore = Dataset('ecore_555', reload=reload)
    modelset = Dataset('modelset', reload=reload, remove_duplicates=True)
    # mar = Dataset('mar-ecore-github', reload=reload)


    datasets = {
        'ecore': ecore,
        'modelset': modelset,
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