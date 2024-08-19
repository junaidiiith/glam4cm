from lang2graph.common import LangGraph


class ArchiMateNxG(LangGraph):
    def __init__(
        self, 
        json_obj: dict, 
        timeout = -1
    ):
        super().__init__()
        self.json_obj = json_obj
        self.timeout = timeout
        self.graph_id = json_obj['identifier'].split('/')[-1]

        self.graph = self.__create_nx_from_file()

        self.set_numbered_labels()
        self.numbered_graph = self.get_numbered_graph()
        
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.numbered_graph.edges())}
        self.idx_to_edge = {idx: edge for idx, edge in enumerate(self.numbered_graph.edges())}



    def __create_nx_from_file(self):
        for node in self.json_obj['elements']:
            self.add_node(node['id'], **node)
        for edge in self.json_obj['relationships']:
            self.add_edge(edge['sourceId'], edge['targetId'], **edge)
