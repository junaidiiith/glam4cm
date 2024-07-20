from lang2graph.common import SEP, LangGraph
import json


EGenericType = 'EGenericType'
EPackage = 'EPackage'
EClass = 'EClass'
EAttribute = 'EAttribute'
EReference = 'EReference'
EEnum = 'EEnum'
EEnumLiteral = 'EEnumLiteral'
EOperation = 'EOperation'
EParameter = 'EParameter'
EDataType = 'EDataType'
GenericNodes = [EGenericType, EPackage]


class EcoreNxG(LangGraph):
    def __init__(
            self, 
            json_obj: dict, 
            use_type=True,
            remove_generic_nodes=True,
        ):
        super().__init__()
        self.graph_id = json_obj.get('ids')
        self.use_type = use_type
        self.remove_generic_nodes = remove_generic_nodes
        self.json_obj = json_obj
        self.graph_type = json_obj.get('model_type')
        self.label = json_obj.get('labels')
        self.is_duplicated = json_obj.get('is_duplicated')
        self.directed = json.loads(json_obj.get('graph')).get('directed')
        self.create_graph()
        self.text = json_obj.get('txt')


    def create_graph(self):
        generic_nodes = list()
        graph = json.loads(self.json_obj['graph'])
        nodes = graph['nodes']
        edges = graph['links']
        for node in nodes:
            self.add_node(node['id'], **node)
            if node['eClass'] in GenericNodes:
                generic_nodes.append(node['id'])
                
        for edge in edges:
            self.add_edge(edge['source'], edge['target'], **edge)
        
        if self.remove_generic_nodes:
            self.remove_nodes_from(generic_nodes)
    
    def get_graph_node_text(self, node):
        data = self.nodes[node]
        node_class = data.get('eClass')
        node_name = data.get('name', '')

        if self.use_type:
            return f'{node_class}({node_name})'

        return node_name


    def find_node_str_upto_distance(self, node, distance=1):
        nodes_with_distance = self.find_nodes_within_distance(
            node, 
            distance=distance
        )
        d2n = {d: set() for _, d in nodes_with_distance}
        for n, d in nodes_with_distance:
            node_text = self.get_graph_node_text(n)
            if node_text:
                d2n[d].add(node_text)
        
        d2n = sorted(d2n.items(), key=lambda x: x[0])
        node_buckets = [f"{SEP}".join(nbs) for _, nbs in d2n]
        path_str = " | ".join(node_buckets)
        
        return path_str


    def get_node_texts(self, distance=1):
        node_texts = []
        for node in self.nodes:
            node_texts.append(
                self.find_node_str_upto_distance(node, distance=distance)
            )
        
        return node_texts
    
        
    def __repr__(self):
        return f'{self.json_obj}\nGraph({self.graph_id}, nodes={self.number_of_nodes()}, edges={self.number_of_edges()})'
