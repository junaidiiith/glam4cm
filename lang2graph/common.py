from abc import abstractmethod
import networkx as nx
from uuid import uuid4
import torch
from tokenization.special_tokens import *

SEP = ' '

class LangGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.id = uuid4().hex
        self.node_label_to_id = {}
        self.id_to_node_label = {}
        self.edge_label_to_id = {}
        self.id_to_edge_label = {}


    @abstractmethod
    def create_graph(self):
        pass

    @abstractmethod
    def get_graph_node_text(self, node):
        pass


    @abstractmethod
    def get_node_texts(self, distance=1):
        pass


    def set_numbered_labels(self):
        self.node_label_to_id = {label: i for i, label in enumerate(self.nodes())}
        self.id_to_node_label = {i: label for i, label in enumerate(self.nodes())}

        self.edge_label_to_id = {label: i for i, label in enumerate(self.edges())}
        self.id_to_edge_label = {i: label for i, label in enumerate(self.edges())}

        self.numbered_graph = self.get_numbered_graph()
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.numbered_graph.edges())}
        self.idx_to_edge = {idx: edge for idx, edge in enumerate(self.numbered_graph.edges())}



    def get_numbered_graph(self) -> nx.DiGraph:
        nodes = [(self.node_label_to_id[i], data) for i, data in list(self.nodes(data=True))]
        edges = [(self.node_label_to_id[i], self.node_label_to_id[j], data) for i, j, data in list(self.edges(data=True))]
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        return graph


    @property
    def enr(self):
        if self.number_of_nodes() == 0:
            return -1
        return self.number_of_edges() / self.number_of_nodes()


    @property
    def edge_index(self):
        edge_index = torch.tensor(list(self.numbered_graph.edges)).t().contiguous()
        return edge_index
    
    def get_edge_id(self, edge):
        return self.edge_label_to_id[edge]

    def get_edge_label(self, edge_id):
        return self.edge_label_to_id[edge_id]

    
    def get_node_id(self, node):
        return self.node_label_to_id[node]
    
    def get_node_label(self, node_id):
        return self.node_label_to_id[node_id]
    


def create_graph_from_edge_index(graph, edge_index: torch.Tensor):
    """
    Create a subgraph from G using only the edges specified in edge_index.
    
    Parameters:
    G (networkx.Graph): The original graph.
    edge_index (torch.Tensor): A tensor containing edge indices.
    
    Returns:
    networkx.Graph: A subgraph of G containing only the edges in edge_index.
    """

    # Add nodes and edges from the edge_index to the subgraph
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(list(graph.numbered_graph.nodes(data=True)))
    subgraph.add_edges_from([(u, v, graph.numbered_graph.edges[u, v]) for u, v in edge_index.t().tolist()])
    for node, data in subgraph.nodes(data=True):
        data = graph.numbered_graph.nodes[node]
        subgraph.nodes[node].update(data)



    subgraph.node_label_to_id = graph.node_label_to_id
    subgraph.id_to_node_label = graph.id_to_node_label
    subgraph.edge_label_to_id = graph.edge_label_to_id
    subgraph.id_to_edge_label = graph.id_to_edge_label
    try:
        assert subgraph.number_of_edges() == edge_index.size(1)
    except AssertionError as e:
        print(f"Number of edges mismatch {subgraph.number_of_edges()} != {edge_index.size(1)}")
        import pickle
        pickle.dump([graph, edge_index], open("subgraph.pkl", "wb"))
        raise e

    return subgraph



def get_node_texts(
        graph: LangGraph, 
        h: int, 
        label='name', 
        use_attributes=False, 
        attribute_labels='attributes'
    ):
    """
    Create node string for each node n in a graph using neighbors of n up to h hops.
    
    Parameters:
    G (networkx.Graph): The graph.
    h (int): The number of hops.
    
    Returns:
    dict: A dictionary where keys are nodes and values are node strings.
    """
    node_texts = {}

    for node in graph.nodes():
        
        node_str = graph.nodes[node][label] if label in graph.nodes[node] else ''
        current_level_nodes = {node}
        all_visited_nodes = {node}

        for _ in range(1, h + 1):
            next_level_nodes = set()
            for n in current_level_nodes:
                neighbors = set(graph.neighbors(n))
                next_level_nodes.update(neighbors - all_visited_nodes)
            all_visited_nodes.update(next_level_nodes)
            if next_level_nodes:
                node_strs = [
                    get_node_name(
                        graph.nodes[i], 
                        label, 
                        use_attributes, 
                        attribute_labels
                    ) for i in sorted(next_level_nodes)
                ]
                node_str += f" {NODE_PATH_SEP} {', '.join(node_strs)}"
            current_level_nodes = next_level_nodes

        node_texts[node] = node_str.strip()

    return node_texts


def get_node_name(
        node_data, 
        label, 
        use_attributes=False,
        attribute_labels='attributes',
    ):
    if use_attributes and attribute_labels in node_data:
        attributes_str = "(" + ', '.join([k for k, _ in node_data[attribute_labels]]) + ")"
    else:
        attributes_str = ''
    node_label = node_data.get(label)
    return f"{node_label} {attributes_str}".strip()


def get_node_data(
    node_data: dict,
    node_label: str,
    model_type: str,
):
    if model_type == 'archimate':
        return get_archimate_node_data(node_data, node_label)
    elif model_type == 'ecore':
        return get_uml_node_data(node_data, node_label)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



def get_edge_data(
    edge_data: dict,
    edge_label: str,
    model_type: str,
):
    if model_type == 'archimate':
        return get_archimate_edge_data(edge_data, edge_label)
    elif model_type == 'ecore':
        return get_uml_edge_data(edge_data, edge_label)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_archimate_node_data(edge_data: dict, node_label: str):
    return edge_data.get(node_label)

def get_uml_node_data(node_data: dict, node_label: str):
    return node_data.get(node_label, '')


def get_archimate_edge_data(edge_data: dict, edge_label: str):
    return edge_data.get(edge_label)


def get_uml_edge_data(edge_data: dict, edge_label: str):
    if edge_label == 'type':
        return get_uml_edge_type(edge_data)
    elif edge_label in edge_data:
        return edge_data[edge_label]
    else:
        raise ValueError(f"Unknown edge label: {edge_label}")


def get_uml_edge_type(edge_data):
    edge_type = edge_data.get('type')
    if edge_type == 'supertype':
        return 'supertype'
    
    containment = edge_data.get('containment')
    if containment:
        return 'containment'

    return 'reference'
