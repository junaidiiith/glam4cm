"""
OntoUML Graph-to-Text Conversion Module

This module provides functionality to convert OntoUML conceptual models from
graph representations to natural language text descriptions. The conversion
process captures the semantic meaning of model elements and their relationships
in a human-readable format.

Key Features:
- Node-to-text conversion with stereotype support
- Edge-to-text conversion for different relationship types
- Path-based text generation for complex model structures
- Distance-based neighborhood exploration
- Triple generation for training data preparation

Supported OntoUML Elements:
- Classes with stereotypes and attributes
- Relationships (reference, containment, supertype)
- Multi-hop path descriptions
- Attribute and stereotype information

Author: Syed Juned Ali
Email: syed.juned.ali@tuwien.ac.at
"""

import networkx as nx
import random
import itertools
from tqdm.auto import tqdm
from constants import *
from common import (
    get_node_neighbours,
    remove_extra_spaces,
    has_neighbours_incl_incoming,
)


def get_node_text_triples(g, distance=1, only_name=False):
    """
    Generate text triples for all nodes in the graph.

    This function creates natural language descriptions for each node in the
    OntoUML graph, including information about the node's type, name, and
    relationships within the specified distance.

    Args:
        g: NetworkX graph representing the OntoUML model
        distance: Maximum distance for neighborhood exploration (default: 1)
        only_name: If True, only return node names without descriptions (default: False)

    Returns:
        List of text descriptions for each node

    Example:
        For a class "Person" with stereotype "Agent", returns:
        "Class Person: Person is_a Agent | Person has_name String"
    """
    node_strings = [get_node_str(g, node, distance) for node in g.nodes]
    node_triples = list()

    for node, node_str in zip(list(g.nodes), node_strings):
        name = g.nodes[node]["name"] if "name" in g.nodes[node] else " reference "
        node_type = g.nodes[node]["type"]

        if only_name:
            prompt_str = f"{name}"
        else:
            prompt_str = f"{node_type} {name}: {node_str}"

        node_triples.append(prompt_str)

    return node_triples


def check_stereotype_relevance(g, n):
    """
    Check if a node has a relevant stereotype that should be included in text.

    Args:
        g: NetworkX graph
        n: Node identifier

    Returns:
        Boolean indicating whether to include stereotype information
    """
    return "use_stereotype" in g.nodes[n] and g.nodes[n]["use_stereotype"]


def process_name_and_steroetype(g, n):
    """
    Process node name and stereotype information for text generation.

    This function combines the node's name with its stereotype (if relevant)
    to create a comprehensive text representation.

    Args:
        g: NetworkX graph
        n: Node identifier

    Returns:
        String combining name and stereotype information

    Example:
        For a class "Person" with stereotype "Agent", returns: "Person Agent"
    """
    string = g.nodes[n]["name"] if g.nodes[n]["name"] != "Null" else ""

    if check_stereotype_relevance(g, n):
        string += f" {g.nodes[n]['stereotype']} "

    return string


def process_node_for_string(g, n, src=True):
    """
    Process a node to generate text representation based on its relationships.

    This function creates text descriptions for a node by considering its
    incoming or outgoing edges and the connected nodes. The direction of
    processing (source vs. target) affects the text structure.

    Args:
        g: NetworkX graph
        n: Node identifier
        src: If True, process as source node; if False, process as target node

    Returns:
        List of text strings describing the node and its relationships

    Example:
        For a class "Person" with containment relationship to "Name":
        - src=True: ["Person contains Name"]
        - src=False: ["Name is_contained_in Person"]
    """
    if g.nodes[n]["type"] == "Class":
        return [process_name_and_steroetype(g, n)]

    strings = list()
    node_str = process_name_and_steroetype(g, n)

    # Get edges based on direction
    edges = list(g.in_edges(n)) if src else list(g.out_edges(n))

    for edge in edges:
        # Get the connected node
        v = edge[0] if src else edge[1]
        v_str = (
            f" {process_edge_for_string(g, edge)} {process_name_and_steroetype(g, v)}"
        )

        # Combine strings based on direction
        if src:
            n_str = v_str + node_str
        else:
            n_str = node_str + v_str

        strings.append(n_str)

    # Remove duplicates and clean up extra spaces
    return list(set(map(remove_extra_spaces, strings)))


def process_edge_for_string(g, e):
    """
    Convert edge information to natural language text.

    This function maps edge types to human-readable relationship descriptions
    using the constants defined in the constants module.

    Args:
        g: NetworkX graph
        e: Edge tuple (u, v)

    Returns:
        String representation of the edge type

    Example:
        - Reference edge: "references"
        - Containment edge: "contains"
        - Supertype edge: "is_a"
    """
    edge_type_s = e_s[g.edges()[e]["type"]]
    return remove_extra_spaces(f" {edge_type_s} ")


def get_triples_from_edges(g, edges=None):
    """
    Generate text triples from graph edges.

    This function creates natural language descriptions of relationships
    between nodes by processing edge information and connected node details.

    Args:
        g: NetworkX graph
        edges: List of edges to process (if None, process all edges)

    Returns:
        List of text triples describing edge relationships

    Example:
        For an edge from "Person" to "Name" with type "containment":
        Returns: ["Person contains Name"]
    """
    if edges is None:
        edges = g.edges()

    triples = []

    for edge in edges:
        u, v = edge
        edge_str = process_edge_for_string(g, edge)

        # Get text representations for both connected nodes
        u_strings, v_strings = (
            process_node_for_string(g, u, src=True),
            process_node_for_string(g, v, src=False),
        )

        # Generate all possible combinations
        for u_str, v_str in itertools.product(u_strings, v_strings):
            pos_triple = u_str + f" {edge_str} " + v_str
            triples.append(remove_extra_spaces(pos_triple))

    return triples


def process_path_string(g, path):
    """
    Convert a path of nodes to a text description.

    This function processes a sequence of connected nodes and creates
    a natural language description of the entire path.

    Args:
        g: NetworkX graph
        path: List of node identifiers representing a path

    Returns:
        Text string describing the path

    Example:
        For path ["Person", "Name", "String"]:
        Returns: "Person contains Name | Name is_a String"
    """
    # Convert path to edge list
    edges = list(zip(path[:-1], path[1:]))

    # Get triples for all edges in the path
    triples = get_triples_from_edges(g, edges)

    # Join triples with separator
    return remove_extra_spaces(f" {SEP} ".join(triples))


def get_triples_from_node(g, n, distance=1):
    """
    Generate text triples for a node considering its neighborhood.

    This function explores the neighborhood of a node up to a specified
    distance and generates text descriptions for all discovered paths.

    Args:
        g: NetworkX graph
        n: Node identifier
        distance: Maximum distance for neighborhood exploration

    Returns:
        List of text triples describing paths from the node

    Example:
        For node "Person" with distance 2:
        - Distance 1: Direct relationships
        - Distance 2: Relationships through intermediate nodes
    """
    triples = list()

    # Temporarily disable stereotype usage to avoid redundancy
    use_stereotype = (
        g.nodes[n]["use_stereotype"] if "use_stereotype" in g.nodes[n] else False
    )
    g.nodes[n]["use_stereotype"] = False

    # Get all nodes within the specified distance
    node_neighbours = get_node_neighbours(g, n, distance)

    for neighbour in node_neighbours:
        # Find all simple paths between the node and its neighbor
        paths = [p for p in nx.all_simple_paths(g, n, neighbour, cutoff=distance)]

        for path in paths:
            triples.append(process_path_string(g, path))

    # Restore original stereotype setting
    g.nodes[n]["use_stereotype"] = use_stereotype

    return triples


def get_node_str(g, n, distance=1):
    """
    Generate a comprehensive text description for a node.

    This function creates a complete text representation of a node by
    considering its relationships, attributes, and neighborhood structure.

    Args:
        g: NetworkX graph
        n: Node identifier
        distance: Maximum distance for neighborhood exploration

    Returns:
        Text string describing the node and its context

    Example:
        For a class "Person" with relationships:
        Returns: "Person contains Name | Person is_a Agent | Name is_a String"
    """
    node_triples = get_triples_from_node(g, n, distance)
    return remove_extra_spaces(f" | ".join(node_triples))


def create_triples_from_graph_edges(graphs):
    """
    Generate text triples from multiple graphs.

    This function processes a collection of graphs and generates text
    triples for all edges across all graphs. Useful for creating
    training datasets from multiple conceptual models.

    Args:
        graphs: List of tuples (graph, metadata) where graph is a NetworkX graph

    Returns:
        List of all text triples from all graphs

    Note:
        This function is designed for batch processing of multiple graphs
        and includes progress tracking for large datasets.
    """
    triples = list()

    for g, _ in tqdm(graphs):
        triples += get_triples_from_edges(g)

    return triples


def mask_graph(
    graph,
    stereotypes_classes,
    mask_prob=0.2,
    use_stereotypes=False,
    use_rel_stereotypes=False,
):
    all_stereotype_nodes = [
        node
        for node in graph.nodes
        if "stereotype" in graph.nodes[node]
        and graph.nodes[node]["stereotype"] in stereotypes_classes
        and has_neighbours_incl_incoming(graph, node)
        and (True if use_rel_stereotypes else graph.nodes[node]["type"] == "Class")
    ]

    assert all(["stereotype" in graph.nodes[node] for node in all_stereotype_nodes]), (
        "All stereotype nodes should have stereotype property"
    )

    total_masked_nodes = int(len(all_stereotype_nodes) * mask_prob)
    masked_nodes = random.sample(all_stereotype_nodes, total_masked_nodes)
    unmasked_nodes = [node for node in all_stereotype_nodes if node not in masked_nodes]

    for node in masked_nodes:
        graph.nodes[node]["masked"] = True
        graph.nodes[node]["use_stereotype"] = False

    for node in unmasked_nodes:
        graph.nodes[node]["masked"] = False
        graph.nodes[node]["use_stereotype"] = use_stereotypes

    assert all(["masked" in graph.nodes[node] for node in all_stereotype_nodes]), (
        "All stereotype nodes should be masked or unmasked"
    )


def mask_graphs(
    graphs,
    stereotypes_classes,
    mask_prob=0.2,
    use_stereotypes=False,
    use_rel_stereotypes=False,
):
    masked, unmasked, total = 0, 0, 0
    # for graph, f_name in tqdm(graphs, desc='Masking graphs'):
    for graph, _ in graphs:
        mask_graph(
            graph,
            stereotypes_classes,
            mask_prob=mask_prob,
            use_stereotypes=use_stereotypes,
            use_rel_stereotypes=use_rel_stereotypes,
        )
        masked += len(
            [
                node
                for node in graph.nodes
                if "masked" in graph.nodes[node] and graph.nodes[node]["masked"]
            ]
        )
        unmasked += len(
            [
                node
                for node in graph.nodes
                if "masked" in graph.nodes[node] and not graph.nodes[node]["masked"]
            ]
        )
        total += len([node for node in graph.nodes if "masked" in graph.nodes[node]])

    ## % of masked nodes upto 2 decimal places
    print(f"Masked {round(masked / total, 2) * 100}%")
    print(f"Unmasked {round(unmasked / total, 2) * 100}%")

    print("Total masked nodes:", masked)
    print("Total unmasked nodes:", unmasked)
