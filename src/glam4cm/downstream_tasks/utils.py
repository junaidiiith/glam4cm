from glam4cm.data_loading.models_dataset import (
    ArchiMateDataset, 
    EcoreDataset,
    OntoUMLDataset
)


dataset_to_metamodel = {
    'modelset': 'ecore',
    'ecore_555': 'ecore',
    'mar-ecore-github': 'ecore',
    'eamodelset': 'ea',
    'ontouml': 'ontouml',
}


def get_metamodel_dataset_type(dataset):
    return dataset_to_metamodel[dataset]


def get_model_dataset_class(dataset_name):
    dataset_type = get_metamodel_dataset_type(dataset_name)
    if dataset_type == 'ea':
        dataset_class = ArchiMateDataset
    elif dataset_type == 'ecore':
        dataset_class = EcoreDataset
    elif dataset_type == 'ontouml':
        dataset_class = OntoUMLDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return dataset_class


def get_models_dataset(dataset_name, **config_params):
    dataset_type = get_metamodel_dataset_type(dataset_name)
    if dataset_type != 'ea' and 'language' in config_params:
        del config_params['language']
    dataset_class = get_model_dataset_class(dataset_name)
    return dataset_class(dataset_name, **config_params)


def get_logging_steps(dataset_size, num_epochs, batch_size):
    """
    Calculate the logging steps based on the dataset size, number of epochs, and batch size.
    """
    num_steps = dataset_size // batch_size
    logging_steps = num_steps * num_epochs // 20
    print(f"Logging steps: {logging_steps}")
    return logging_steps