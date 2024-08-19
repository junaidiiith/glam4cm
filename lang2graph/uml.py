from lang2graph.common import LangGraph
import json

from pyecore.resources import ResourceSet, URI
from lang2graph.utils import (
    run_with_timeout
)
import os



REFERENCE = 'reference'
SUPERTYPE = 'supertype'

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
            timeout = -1
        ):
        super().__init__()
        self.xmi = json_obj.get('xmi')
        self.graph_id = json_obj.get('ids')
        self.timeout = timeout
        self.json_obj = json_obj
        self.graph_type = json_obj.get('model_type')
        self.label = json_obj.get('labels')
        self.is_duplicated = json_obj.get('is_duplicated')
        self.directed = json.loads(json_obj.get('graph')).get('directed')
        self.__create_nx_from_xmi()
        self.set_numbered_labels()
        self.numbered_graph = self.get_numbered_graph()
        
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.numbered_graph.edges())}
        self.idx_to_edge = {idx: edge for idx, edge in enumerate(self.numbered_graph.edges())}

        self.text = json_obj.get('txt')
    
    
    def __create_nx_from_file(self, file_name):
        references, supertypes = get_ecore_data(file_name)
        for class_name, class_references in references.items():
            self.add_node(class_name, name=class_name)
            
            for class_reference in class_references.values():
                if class_reference['type'] == 'EReference':
                    self.add_edge(
                        class_name,
                        class_reference['eType'], 
                        name=class_reference['name'], 
                        type = REFERENCE,
                        containment=class_reference['containment']
                    )
                
                elif class_reference['type'] == 'EAttribute':
                    if 'attributes' not in self.nodes[class_name]:
                        self.nodes[class_name]['attributes'] = list()

                    self.nodes[class_name]['attributes'].append(
                        (class_reference['name'], class_reference['eType'])
                    )

        
        for class_name, class_super_types in supertypes.items():
            for supertype_name, class_super_type in class_super_types.items():
                self.nodes[supertype_name]['abstract'] = True
                self.add_edge(
                    class_name, 
                    class_super_type['name'], 
                    type = SUPERTYPE
                )


    def __create_nx_from_xmi(self):
        with open('temp.xmi', 'w') as f:
            f.write(self.xmi)
        
        if self.timeout != -1:
            nxg = run_with_timeout(
                self.__create_nx_from_file, 
                args=('temp.xmi',), 
                timeout_duration=self.timeout
            )
        else:
            nxg = self.__create_nx_from_file('temp.xmi')
        
        os.remove('temp.xmi')
        return nxg


    def __repr__(self):
        return f'{self.json_obj}\nGraph({self.graph_id}, nodes={self.number_of_nodes()}, edges={self.number_of_edges()})'



def get_resource_from_file(file_name):
    rset = ResourceSet()
    resource = rset.get_resource(URI(file_name))
    return resource


def get_supertype_features(super_type):
    feats = {feat.name for feat in super_type.eAllStructuralFeatures()}
    for super_super_type in super_type.eAllSuperTypes():
        feats = feats.union(get_supertype_features(super_super_type))
    return feats


def get_ereferences(classifier):
    ereferences = {
        feat.name: {
            "name": feat.name,
            "type": type(feat).__name__, 
            "eType": feat.eType.name, 
            "containment": getattr(feat, 'containment', None)
        }
        for feat in classifier.eAllStructuralFeatures()
    }

    super_type_feats = set()
    for super_type in classifier.eAllSuperTypes():
        super_type_feats = super_type_feats.union(get_supertype_features(super_type))
    
    for feat_name in super_type_feats:
        if feat_name in ereferences:
            del ereferences[feat_name]
        
    return ereferences


def get_esupertypes(classifier):
    esupertypes = {
        supertype.name : {
            "name": supertype.name,
            "type": type(supertype).__name__, 
        }
        
        for supertype in classifier.eAllSuperTypes()
    }
    return esupertypes


def get_ecore_data(file_name):
    resource = get_resource_from_file(file_name)
    references, supertypes = dict(), dict()
    for mm_root in resource.contents:
        for classifier in mm_root.eClassifiers:
            if type(classifier).__name__ == 'EClass':
                ereferences = get_ereferences(classifier)
                esupertypers = get_esupertypes(classifier)
                references[classifier.name] = ereferences
                supertypes[classifier.name] = esupertypers

    return references, supertypes
