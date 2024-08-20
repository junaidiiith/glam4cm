from data_loading.models_dataset import (
    ArchiMateModelDataset, 
    EcoreModelDataset
)


dataset_to_metamodel = {
    'modelset': 'ecore',
    'ecore_555': 'ecore',
    'mar-ecore-github': 'ecore',
    'eamodelset': 'ea'
}


def get_metamodel_dataset_type(dataset):
    return dataset_to_metamodel[dataset]


def get_model_dataset_class(dataset_name):
    dataset_type = get_metamodel_dataset_type(dataset_name)
    if dataset_type == 'ea':
        dataset_class = ArchiMateModelDataset
    elif dataset_type == 'ecore':
        dataset_class = EcoreModelDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return dataset_class


def get_models_dataset(dataset_name, **config_params):
    dataset_class = get_model_dataset_class(dataset_name)
    return dataset_class(dataset_name, **config_params)