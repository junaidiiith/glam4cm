class GraphMetadata:
    def __init__(self, model_type):
        self.type = model_type

    @property
    def node_label(self):
        return self.node.get('label')
    
    @property
    def node_cls(self):
        return self.node.get('cls')

    @property
    def node_attributes(self):
        return self.node.get('attributes')
    
    @property
    def edge_label(self):
        return self.edge.get('label')
    
    @property
    def edge_cls(self):
        return self.edge.get('cls')
    
    @property
    def graph_cls(self):
        return self.graph.get('cls')
        
    @property
    def graph_label(self):
        return self.graph.get('label')



class EcoreMetaData(GraphMetadata):
    def __init__(self):
        super().__init__('ecore')
        self.node = {
            "label": "name",
            "cls": "abstract",
            "attributes": "attributes"
        }
        self.edge = {
            "label": "name",
            "cls": "type"
        }
        self.graph = {
            "label": "text",
            "cls": "label"
        }



class ArchimateMetaData(GraphMetadata):
    def __init__(self):
        super().__init__('archimate')
        self.node = {
            "label": "name",
            "cls": ["type", "layer"],
        }
        self.edge = {
            "cls": "type"
        }

        self.graph = {}