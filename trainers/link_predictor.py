import torch
from data_loading.loader import get_data_loaders


class LinkPredictor:
    def __init__(
        self,
        model,
        dataset,
    ):
        self.model = model
        self.dataset = dataset
        self.train_loader, self.test_loader = get_data_loaders(self.dataset)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.BCEWithLogitsLoss()