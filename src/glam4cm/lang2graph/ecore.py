"""
Ecore Language-to-Graph Conversion Module

This module provides functionality to convert Ecore (Eclipse Modeling Framework)
metamodels from XML/XMI format to graph representations. The conversion process
parses Ecore model definitions and creates NetworkX graphs that can be used
for machine learning tasks.

Key Features:
- XMI/XML parsing for Ecore models
- Graph construction with node and edge attributes
- Support for various Ecore elements (EClass, EAttribute, EReference, etc.)
- Metadata extraction and graph labeling
- Duplicate detection and graph validation

Supported Ecore Elements:
- EClass: Class definitions with attributes and operations
- EAttribute: Class attributes with types
- EReference: Relationships between classes
- EPackage: Package organization
- EEnum: Enumeration types
- EOperation: Class methods and operations

Author: Syed Juned Ali
Email: syed.juned.ali@tuwien.ac.at
"""

import xmltodict
from glam4cm.lang2graph.common import LangGraph
import json
from glam4cm.tokenization.utils import doc_tokenizer
from glam4cm.settings import logger

# =============================================================================
# ECORE CONSTANTS
# =============================================================================

# Edge type constants for Ecore models
REFERENCE = "reference"  # Reference relationship between classes
SUPERTYPE = "supertype"  # Inheritance relationship
CONTAINMENT = "containment"  # Composition relationship

# Ecore element type identifiers
EGenericType = "EGenericType"  # Generic type information
EPackage = "EPackage"  # Package container
EClass = "EClass"  # Class definition
EAttribute = "EAttribute"  # Class attribute
EReference = "EReference"  # Class reference
EEnum = "EEnum"  # Enumeration type
EEnumLiteral = "EEnumLiteral"  # Enumeration value
EOperation = "EOperation"  # Class operation
EParameter = "EParameter"  # Operation parameter
EDataType = "EDataType"  # Data type definition

# Generic node types that don't require special processing
GenericNodes = [EGenericType, EPackage]


class EcoreNxG(LangGraph):
    """
    Ecore to NetworkX Graph converter.

    This class extends the base LangGraph class to provide specialized
    functionality for converting Ecore metamodels to NetworkX graph
    representations. The conversion process preserves the semantic
    structure of the Ecore model while creating a graph suitable for
    machine learning tasks.

    Attributes:
        xmi: Raw XMI/XML content of the Ecore model
        graph_id: Unique identifier for the graph
        json_obj: Parsed JSON representation of the model
        graph_type: Type of the model (e.g., 'ecore')
        label: Labels associated with the model
        is_duplicated: Flag indicating if this is a duplicate model
        directed: Whether the graph is directed

    Example:
        # Create graph from Ecore JSON
        ecore_graph = EcoreNxG(ecore_json_data)

        # Access graph properties
        print(f"Nodes: {ecore_graph.number_of_nodes()}")
        print(f"Edges: {ecore_graph.number_of_edges()}")
    """

    def __init__(self, json_obj: dict):
        """
        Initialize the Ecore graph converter.

        Args:
            json_obj: Dictionary containing Ecore model information including
                     XMI content, metadata, and graph structure

        The initialization process:
        1. Extracts model metadata and content
        2. Parses the XMI/XML content
        3. Creates the graph structure
        4. Sets up node and edge labels
        """
        super().__init__()

        # Extract model metadata
        self.xmi = json_obj.get("xmi")
        self.graph_id = json_obj.get("ids")
        self.json_obj = json_obj
        self.graph_type = json_obj.get("model_type")
        self.label = json_obj.get("labels")
        self.is_duplicated = json_obj.get("is_duplicated")

        # Parse graph structure information
        graph_info = json.loads(json_obj.get("graph"))
        self.directed = graph_info.get("directed")

        # Note: Text tokenization is commented out but available
        # self.text = doc_tokenizer(json_obj.get('txt'))

        # Build the graph structure
        self.__create_graph()

        # Set up numbered labels for machine learning tasks
        self.set_numbered_labels()

    def __create_graph(self):
        """
        Create the graph structure from Ecore model elements.

        This method:
        1. Parses the XMI/XML content to extract Ecore elements
        2. Creates nodes for each classifier (class, enum, etc.)
        3. Establishes edges for relationships between elements
        4. Adds attributes and metadata to nodes and edges

        The graph construction follows Ecore metamodel semantics:
        - EClass elements become nodes with class information
        - EReference elements become edges with relationship types
        - EAttribute elements are stored as node attributes
        - Inheritance relationships create supertype edges
        """
        # Parse XMI content to extract Ecore elements
        model = xmltodict.parse(self.xmi)
        eclassifiers, _ = get_eclassifiers(model)

        # Create a mapping of classifier names to their information
        classifier_nodes = dict()
        for eclassifier in eclassifiers:
            eclassifier_info = get_eclassifier_info(eclassifier)
            classifier_nodes[eclassifier_info["name"]] = eclassifier_info

        # Extract relationship information between classifiers
        references = get_connections(classifier_nodes)

        # Create nodes for each classifier
        for classifier_name, classifier_info in classifier_nodes.items():
            # Extract structural features (attributes and references)
            structural_features = classifier_info.get("structural_features", [])
            attributes = list()

            # Process attributes
            for f in structural_features:
                if f["type"] == "ecore:EAttribute":
                    name = f["name"]
                    attributes.append(name)

            # Add node to graph with metadata
            self.add_node(
                classifier_name,
                name=classifier_name,
                attributes=attributes,
                abstract=classifier_info["abstract"],
            )

        # Create edges for relationships
        for edge in references:
            src, dest = edge["source"], edge["target"]
            name = edge["name"] if "name" in edge else ""
            self.add_edge(src, dest, name=name, type=edge["type"])

        # Set default abstract flag for nodes that don't have it
        for node in self.nodes:
            if (
                "abstract" not in self.nodes[node]
                or self.nodes[node]["abstract"] is None
            ):
                self.nodes[node]["abstract"] = False

        # Log graph creation summary
        logger.info(
            f"Graph {self.graph_id} created with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges"
        )

    def __str__(self):
        """String representation of the graph."""
        return self.__repr__()

    def __repr__(self):
        """
        Detailed string representation of the Ecore graph.

        Returns:
            String containing graph statistics and edge type counts

        Example:
            "EcoreNxG(graph_123, nodes=15, edges=20, references=12, containment=5, supertypes=3)"
        """
        # Count edges by type
        reference_edges = [
            edge for edge in self.edges if self.edges[edge]["type"] == REFERENCE
        ]
        containment_edges = [
            edge for edge in self.edges if self.edges[edge]["type"] == CONTAINMENT
        ]
        supertype_edges = [
            edge for edge in self.edges if self.edges[edge]["type"] == SUPERTYPE
        ]

        return (
            f"EcoreNxG({self.graph_id}, nodes={self.number_of_nodes()}, "
            f"edges={self.number_of_edges()}, references={len(reference_edges)}, "
            f"containment={len(containment_edges)}, supertypes={len(supertype_edges)})"
        )


def get_eclassifiers(json_obj):
    """
    Extract EClassifier elements from parsed Ecore model.

    This function recursively searches through the parsed XML/JSON structure
    to find all EClassifier elements, which represent the main building
    blocks of Ecore models (classes, enums, data types, etc.).

    Args:
        json_obj: Parsed XML/JSON object from xmltodict

    Returns:
        Tuple of (eclassifiers, package_name) where:
        - eclassifiers: List of classifier dictionaries
        - package_name: Name of the containing package

    Note:
        This function handles nested package structures and extracts
        classifiers from all levels of the model hierarchy.
    """

    def get_eclassifiers_util(json_obj, classifiers: list):
        """
        Recursive utility function to find EClassifier elements.

        Args:
            json_obj: Current JSON object to search
            classifiers: List to accumulate found classifiers
        """
        for key, value in json_obj.items():
            if key == "eClassifiers":
                # Handle both single classifier and list of classifiers
                if isinstance(value, dict):
                    value = [value]
                classifiers.extend(value)
            elif isinstance(value, dict):
                # Recursively search nested objects
                get_eclassifiers_util(value, classifiers)
            elif isinstance(value, list):
                # Recursively search list items
                for item in value:
                    if isinstance(item, dict):
                        get_eclassifiers_util(item, classifiers)

    # Initialize classifiers list and find package name
    classifiers = []
    package_name = None

    # Extract package name if available
    if "ePackage" in json_obj:
        package_info = json_obj["ePackage"]
        if isinstance(package_info, dict) and "name" in package_info:
            package_name = package_info["name"]

    # Find all classifiers
    get_eclassifiers_util(json_obj, classifiers)

    return classifiers, package_name


def get_eclassifier_info(eclassifier):
    """
    Extract information from an EClassifier element.

    This function processes individual EClassifier elements to extract
    their properties, structural features, and metadata.

    Args:
        eclassifier: Dictionary representing an EClassifier element

    Returns:
        Dictionary containing classifier information:
        - name: Name of the classifier
        - type: Type of the classifier (EClass, EEnum, etc.)
        - abstract: Whether the classifier is abstract
        - structural_features: List of attributes and references
        - operations: List of operations/methods

    Example:
        For an EClass "Person" with attributes:
        {
            'name': 'Person',
            'type': 'EClass',
            'abstract': False,
            'structural_features': [{'name': 'name', 'type': 'ecore:EAttribute'}]
        }
    """
    classifier_info = {}

    # Extract basic properties
    classifier_info["name"] = eclassifier.get("name", "Unknown")
    classifier_info["type"] = eclassifier.get("@xsi:type", "Unknown")
    classifier_info["abstract"] = eclassifier.get("@abstract", False)

    # Extract structural features (attributes and references)
    structural_features = []
    if "eStructuralFeatures" in eclassifier:
        features = eclassifier["eStructuralFeatures"]
        if isinstance(features, dict):
            features = [features]

        for feature in features:
            if isinstance(feature, dict):
                feature_info = {
                    "name": feature.get("name", ""),
                    "type": feature.get("@xsi:type", ""),
                    "lowerBound": feature.get("@lowerBound", "1"),
                    "upperBound": feature.get("@upperBound", "1"),
                }
                structural_features.append(feature_info)

    classifier_info["structural_features"] = structural_features

    # Extract operations (methods)
    operations = []
    if "eOperations" in eclassifier:
        ops = eclassifier["eOperations"]
        if isinstance(ops, dict):
            ops = [ops]

        for op in ops:
            if isinstance(op, dict):
                op_info = {"name": op.get("name", ""), "parameters": []}

                # Extract operation parameters
                if "eParameters" in op:
                    params = op["eParameters"]
                    if isinstance(params, dict):
                        params = [params]

                    for param in params:
                        if isinstance(param, dict):
                            param_info = {
                                "name": param.get("name", ""),
                                "type": param.get("@xsi:type", ""),
                            }
                            op_info["parameters"].append(param_info)

                operations.append(op_info)

    classifier_info["operations"] = operations

    return classifier_info


def get_connections(classifier_nodes):
    """
    Extract connection information between classifiers.

    This function analyzes the structural features of classifiers to
    identify relationships between them, such as references, containment,
    and inheritance.

    Args:
        classifier_nodes: Dictionary mapping classifier names to their information

    Returns:
        List of connection dictionaries, each containing:
        - source: Source classifier name
        - target: Target classifier name
        - type: Type of relationship (reference, containment, supertype)
        - name: Name of the relationship (optional)

    Example:
        [
            {'source': 'Person', 'target': 'Name', 'type': 'containment', 'name': 'hasName'},
            {'source': 'Person', 'target': 'Agent', 'type': 'supertype', 'name': ''}
        ]
    """
    connections = []

    for classifier_name, classifier_info in classifier_nodes.items():
        structural_features = classifier_info.get("structural_features", [])

        for feature in structural_features:
            if feature["type"] == "ecore:EReference":
                # Extract reference information
                ref_name = feature.get("name", "")
                ref_type = feature.get("@xsi:type", "")

                # Determine relationship type based on reference properties
                if "containment" in feature and feature["containment"] == "true":
                    rel_type = CONTAINMENT
                elif "eType" in feature:
                    # Check if this is a supertype relationship
                    etype = feature["eType"]
                    if isinstance(etype, dict) and "name" in etype:
                        target_name = etype["name"]
                        if target_name in classifier_nodes:
                            connections.append(
                                {
                                    "source": classifier_name,
                                    "target": target_name,
                                    "type": REFERENCE,
                                    "name": ref_name,
                                }
                            )
                else:
                    rel_type = REFERENCE

                # Add containment relationships
                if rel_type == CONTAINMENT and "eType" in feature:
                    etype = feature["eType"]
                    if isinstance(etype, dict) and "name" in etype:
                        target_name = etype["name"]
                        if target_name in classifier_nodes:
                            connections.append(
                                {
                                    "source": classifier_name,
                                    "target": target_name,
                                    "type": CONTAINMENT,
                                    "name": ref_name,
                                }
                            )

    return connections
