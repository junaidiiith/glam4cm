from data_loading.dataset import GraphDataset
from torch_geometric.loader import DataLoader


def get_train_test_dataloaders(
        dataset: GraphDataset, 
        train_idx: list, 
        test_idx: list, 
        batch_size = 32
    ):
    train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dataloaders = {
        'train': train_loader,
        'test': test_loader
    }

    return dataloaders


def get_data_loaders(dataset: GraphDataset, tr = 0.8, batch_size = 32):
    train_idx, test_idx = dataset.get_train_test_split(tr=tr)
    return get_train_test_dataloaders(
        dataset, 
        train_idx, 
        test_idx, 
        batch_size
    )
    

def get_k_fold_data_loaders(dataset: GraphDataset, k=10, batch_size=32):
    for train_idx, test_idx in dataset.k_fold_split(k=k):
        yield get_train_test_dataloaders(dataset, train_idx, test_idx, batch_size)