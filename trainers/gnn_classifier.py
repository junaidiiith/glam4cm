from data_loading.dataset import Dataset, get_embedding_graph_dataset
from embeddings.common import Embedder
from settings import device
from torch_geometric.loader import DataLoader
import torch
from models.gnn_layers import GraphSAGE
from torch.functional import F


class GNNTrainer:
    def __init__(
            self,
            model : torch.nn.Module,
            train_dataset,
            test_dataset,
            batch_size=32,
            lr=0.01,
            weight_decay=5e-4,
        ):

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )

        self.test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )


    def train(self):
        avg_loss = 0
        self.model.train()
        for data in self.train_dataloader:
            data = data.to(device)
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()
        
        return avg_loss / len(self.train_dataloader)


    def test(self):
        self.model.eval()
        correct = 0
        for data in self.test_dataloader:
            data = data.to(device)
            out = self.model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
        return correct / len(self.test_dataloader.dataset)


def run(
        dataset: Dataset, 
        embedder: Embedder,
        batch_size=32,
        hidden_dim=64,
        num_layers=3,
        num_epochs=100,
    ):
    graph_dataset = get_embedding_graph_dataset(embedder)
    train_idx, test_idx = dataset.get_train_test_split()

    train_dataset = [graph_dataset[i] for i in train_idx]
    test_dataset = [graph_dataset[i] for i in test_idx]


    # Training the model
    model = GraphSAGE(
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
