import torch
import json
import os
from data_loading.models_dataset import ModelDataset
from embeddings.bert import BertEmbedder
from lang2graph.uml import EcoreNxG
from lang2graph.common import (
    create_graph_from_edge_index, 
    get_edge_texts, 
    get_node_texts
)

from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data


from settings import BERT_MODEL
embedder = BertEmbedder(model_name=BERT_MODEL)


class TorchGraph:
    def __init__(
            self, 
            graph: EcoreNxG, 
            save_dir: str,
            distance = 1,
            lptr=0.2,
            use_neg_samples=False,
            neg_samples_ratio=1,
            reload=False,
        ):
        self.reload = reload
        self.graph = graph
        self.distance = distance
        self.add_negative_train_samples = use_neg_samples
        self.neg_sampling_ratio = neg_samples_ratio
        self.lptr = lptr
        self.save_dir = save_dir
        self.process_graph()
    

    def process_graph(self):
        if not self.load_pyg_data() or self.reload:
            self.data = self.get_pyg_data()
            self.validate_data()
                    
        self.save()

    
    def get_pyg_data(self):
        transform = RandomLinkSplit(
            num_val=0, 
            num_test=self.lptr, 
            add_negative_train_samples=self.add_negative_train_samples,
            neg_sampling_ratio=self.neg_sampling_ratio,
            split_labels=True
        )

        train_data, _, test_data = transform(Data(
            edge_index=self.graph.edge_index, 
            num_nodes=self.graph.number_of_nodes()
        ))
        edge_index = train_data.edge_index
        subgraph = create_graph_from_edge_index(self.graph, edge_index)

        node_texts = get_node_texts(subgraph, self.distance)
        node_embeddings = embedder.embed(list(node_texts.values()))

        edge_texts = get_edge_texts(subgraph)

        edge_embeddings = embedder.embed(list(edge_texts.values()))


        data = Data(
            x=node_embeddings,
            edge_index=edge_index,
            edge_attr=edge_embeddings,
            train_data=train_data,
            test_data=test_data,
        )

        return data
    

    def validate_data(self):
        pass

    @property
    def name(self):
        return '.'.join(self.graph.graph_id.replace('/', '_').split('.')[:-1])


    @property
    def save_idx(self):
        path = os.path.join(self.save_dir, f'{self.graph.id}')
        if embedder.finetuned:
            path = f'{path}_finetuned'
        return path


    def save_to_mapping(self):
        graph_embedding_file_map = dict()
        fp = f'{self.save_dir}/mapping.json'
        if os.path.exists(fp):
            graph_embedding_file_map = json.load(open(fp, 'r'))
        else:
            graph_embedding_file_map = dict()
        
        graph_embedding_file_map[self.name] = self.graph.id
        json.dump(graph_embedding_file_map, open(fp, 'w'), indent=4)


    def load_pyg_data(self):

        if os.path.exists(self.save_idx):
            self.save_to_mapping()
            self.data = torch.load(f"{self.save_idx}/data.pt")
            return True

        return False


    def save(self):
        os.makedirs(self.save_idx, exist_ok=True)
        torch.save(self.data, f"{self.save_idx}/data.pt")
        self.save_to_mapping()



def get_models_dataset(
        dataset_name, 
        reload=False, 
        remove_duplicates=False, 
        use_type=False, 
    ):
    return ModelDataset(
        dataset_name, 
        reload=reload, 
        remove_duplicates=remove_duplicates, 
        use_type=use_type, 
    )
    