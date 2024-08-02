from data_loading.data import LinkPredictionDataLoader
from data_loading.graph_dataset import GraphDataset
from models.gnn_layers import GNNLinkPredictor
from torch.optim import Adam
from settings import device
import torch


class GNNLinkPredictorTrainer:
    def __init__(
            self,
            model: GNNLinkPredictor,
            dataset: GraphDataset,
            batch_size=32,
            lr=0.01,
            weight_decay=5e-4,
            use_edge_classifier=False,
            use_link_predictor=False
        ):

        assert use_edge_classifier or use_link_predictor, 'At least one of the classifiers should be used'
        
        self.use_edge_classifier = use_edge_classifier
        self.use_link_predictor = use_link_predictor
        
        self.model = model
        self.model.to(device)
        self.dataloader = LinkPredictionDataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.optimizer = Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )


    def step(self, data):
        data = data.to(device)
        self.optimizer.zero_grad()
        edge_attr = data.edge_attr[data.train_edge_idx]
        pos_edge_index = data.train_pos_edge_label_index
        neg_edge_index = data.train_neg_edge_label_index
        edge_classes = data.edge_classes[data.train_edge_idx]

        h = self.model(data.x, pos_edge_index, edge_attr)
        
        loss = 0
        if self.use_link_predictor:
            loss += self.model.link_predictor_loss(h, pos_edge_index, neg_edge_index)        
        if self.use_edge_classifier:
            loss += self.model.edge_classifier_loss(h, pos_edge_index, edge_classes)
            
        loss.backward()
        self.optimizer.step()
        return loss


    def train(self):
        avg_loss = 0
        self.model.train()
        for data in self.dataloader:
            loss = self.step(data)
            avg_loss += loss.item()
        
        return avg_loss / len(self.dataloader)


    def test(self):
        self.model.eval()
        lp_correct, ec_correct = 0, 0
        for data in self.dataloader:
            data = data.to(device)

            edge_attr = data.edge_attr[data.test_edge_idx]

            pos_edge_index = data.test_pos_edge_label_index
            neg_edge_index = data.test_neg_edge_label_index
            edge_classes = data.edge_classes[data.test_edge_idx]

            h = self.model(data.x, pos_edge_index, edge_attr)
            
            if self.use_link_predictor:
                pos_pred, neg_pred = self.model.get_link_predictions(h, pos_edge_index, neg_edge_index)
                pred = torch.cat([pos_pred, neg_pred])
                pred = pred > 0

                edge_label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
                lp_correct += pred.eq(edge_label).sum().item()
            

            if self.use_edge_classifier:
                pred = self.model.get_edge_classifier_predictions(h, pos_edge_index)
                pred_classes = pred.argmax(dim=-1)
                ec_correct += pred_classes.eq(edge_classes).sum().item()


        if self.use_link_predictor:
            lp_acc = lp_correct / len(self.dataloader.dataset)
        else:
            lp_acc = None
        
        if self.use_edge_classifier:
            ec_acc = ec_correct / len(self.dataloader.dataset)
        else:
            ec_acc = None
        
        return lp_acc, ec_acc
        

    def train_epochs(self, num_epochs):
        for epoch in range(num_epochs):
            loss = self.train()
            lp_acc, ec_acc = self.test()
            print(f'Epoch {epoch}, Loss: {loss}, LP Acc: {lp_acc}, EC Acc: {ec_acc}')

