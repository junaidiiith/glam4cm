from abc import abstractmethod
import networkx as nx
from uuid import uuid4
import torch

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
    



def get_uml_edge_type(edge_data):

    # Reference = 2
    # Containment = 1
    # Supertype = 0

    if edge_data['type'] == 'supertype':
        return 0
    if edge_data['containment']:
        return 1
    return 2



def create_graph_from_edge_index(graph, edge_index):
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



def get_node_texts(graph, h: int):
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
        node_str = f"{node}"
        current_level_nodes = {node}
        all_visited_nodes = {node}

        for _ in range(1, h + 1):
            next_level_nodes = set()
            for n in current_level_nodes:
                neighbors = set(graph.neighbors(n))
                next_level_nodes.update(neighbors - all_visited_nodes)
            all_visited_nodes.update(next_level_nodes)
            if next_level_nodes:
                node_strs = [graph.id_to_node_label[i] for i in sorted(next_level_nodes)]
                node_str += f" -> {', '.join(map(str, node_strs))}"
            current_level_nodes = next_level_nodes

        node_texts[node] = node_str

    return node_texts


def get_edge_texts(graph, use_edge_types=False):
    """
    Create edge string for each edge in a graph.
    
    Parameters:
    G (networkx.Graph): The graph.
    
    Returns:
    dict: A dictionary where keys are edges and values are edge strings.
    """
    edge_texts = {}

    for u, v, data in graph.edges(data=True):
        if use_edge_types:
            edge_texts[(u, v)] = f"{graph.node_label_to_id[u]} - {get_uml_edge_type(data)} - {graph.node_label_to_id[v]}"
        else:
            edge_texts[(u, v)] = f"{graph.node_label_to_id[u]} - {graph.node_label_to_id[v]}"


    assert len(edge_texts) == graph.number_of_edges(), f"#Edges text mismatch {len(edge_texts)} != {graph.number_of_edges()}"
    return edge_texts


def get_uml_edge_type(edge_data):

    # Reference = 0
    # Containment = 1
    # Supertype = 2

    if edge_data['type'] == "supertype":
        return 2
    if edge_data['containment']:
        return 1
    return 0

