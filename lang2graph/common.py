from abc import abstractmethod
import networkx as nx
from uuid import uuid4

SEP = ' '



class LangGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.id = uuid4().hex

    @abstractmethod
    def create_graph(self):
        pass

    @abstractmethod
    def get_graph_node_text(self, node):
        pass

    @abstractmethod
    def find_node_str_upto_distance(self, node, distance=1):
        pass

    @abstractmethod
    def get_node_texts(self, distance=1):
        pass

    def find_nodes_within_distance(self, n, distance=1):
        visited = {n: 0}
        queue = [(n, 0)]
        
        while queue:
            node, d = queue.pop(0)
            if d == distance:
                continue
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    visited[neighbor] = d+1
                    queue.append((neighbor, d+1))
        
        visited = sorted(visited.items(), key=lambda x: x[1])
        return visited
