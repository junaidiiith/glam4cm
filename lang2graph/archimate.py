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

        self.graph = self.__create_nx_from_file()


    def __create_nx_from_file(self):
        for node in self.json_obj['elements']:
            self.add_node(node['id'], **node)
        for edge in self.json_obj['relationships']:
            self.add_edge(edge['sourceId'], edge['targetId'], **edge)

