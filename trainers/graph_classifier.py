from data_loading.graph_dataset import GraphDataset
from data_loading.models_dataset import ModelDataset
from embeddings.common import Embedder
from settings import device
import torch
from models.gnn_layers import GNNModel
from torch.functional import F



class GNNTrainer:
    def __init__(
            self,
            model : torch.nn.Module,
            dataset: GraphDataset,
            batch_size=32,
            tr=0.8,
            lr=0.01,
            weight_decay=5e-4,
        ):

        self.model = model
        self.model.to(device)
        self.dataloaders = list()
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    
    def step(self, data):
        data = data.to(device)
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def train(self):
        avg_loss = 0
        self.model.train()
        for data in self.dataloaders['train']:
            loss = self.step(data)
            avg_loss += loss.item()
        
        return avg_loss / len(self.dataloaders['train'])


    def test(self):
        self.model.eval()
        correct = 0
        for data in self.dataloaders['test']:
            data = data.to(device)
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
        return correct / len(self.dataloaders['test'].dataset)
    

    def train_epochs(self, num_epochs):
        for epoch in range(num_epochs):
            loss = self.train()
            acc = self.test()
            print(f'Epoch {epoch+1}/{num_epochs}: Loss: {loss:.4f}, Acc: {acc:.4f}')
    

def run(
        dataset: ModelDataset, 
        embedder: Embedder,
        batch_size=32,
        hidden_dim=64,
        num_layers=3,
        num_epochs=100,
    ):
    graph_dataset: GraphDataset = list()
    train_idx, test_idx = dataset.get_train_test_split()

    train_dataset = [graph_dataset[i] for i in train_idx]
    test_dataset = [graph_dataset[i] for i in test_idx]


    # Training the model
    model = GNNModel(
        input_dim=graph_dataset.num_features,
        hidden_dim=hidden_dim, 
        output_dim=graph_dataset.num_classes,
        num_layers=num_layers,
    ).to(device)
    
    
    trainer = GNNTrainer(
        model, 
        train_dataset, 
        test_dataset, 
        batch_size=batch_size
    )

    for epoch in range(num_epochs):
        loss = trainer.train()
        acc = trainer.test()
        print(f'Epoch {epoch+1}/{num_epochs}: Loss: {loss:.4f}, Acc: {acc:.4f}')
