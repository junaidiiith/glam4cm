import networkx as nx
import random
import itertools
from tqdm.auto import tqdm
from constants import *
from common import (
    get_node_neighbours,
    remove_extra_spaces,
    has_neighbours_incl_incoming
)


def get_node_text_triples(g, distance=1, only_name=False):
    node_strings = [get_node_str(g, node, distance) for node in g.nodes]
    node_triples = list()
    for node, node_str in zip(list(g.nodes), node_strings):
        name = g.nodes[node]['name'] if 'name' in g.nodes[node] else " reference "
        node_type = g.nodes[node]['type']
        prompt_str = f"{node_type} {name}: {node_str}" if not only_name else f"{name}"
        node_triples.append(prompt_str)
    return node_triples


def check_stereotype_relevance(g, n):
    return 'use_stereotype' in g.nodes[n] and g.nodes[n]['use_stereotype']


def process_name_and_steroetype(g, n):
    string = g.nodes[n]['name'] if g.nodes[n]['name'] != "Null" else ""
    string += f' {g.nodes[n]["stereotype"]} ' if check_stereotype_relevance(g, n) else ""
        
    return string


def process_node_for_string(g, n, src=True):
    if g.nodes[n]['type'] == 'Class':
        return [process_name_and_steroetype(g, n)]
        
    strings = list()
    node_str = process_name_and_steroetype(g, n)
    edges = list(g.in_edges(n)) if src else list(g.out_edges(n))
    for edge in edges:
        v = edge[0] if src else edge[1]
        v_str = f" {process_edge_for_string(g, edge)} {process_name_and_steroetype(g, v)}"
        n_str = v_str + node_str if src else node_str + v_str
        strings.append(n_str)
    return list(set(map(remove_extra_spaces, strings)))


def process_edge_for_string(g, e):
    edge_type_s = e_s[g.edges()[e]['type']]
    return remove_extra_spaces(f" {edge_type_s} ")


def get_triples_from_edges(g, edges=None):
    if edges is None:
        edges = g.edges()
    triples = []
    for edge in edges:
        u, v = edge
        edge_str = process_edge_for_string(g, edge)
        u_strings, v_strings = process_node_for_string(g, u, src=True), process_node_for_string(g, v, src=False)
        for u_str, v_str in itertools.product(u_strings, v_strings):
            pos_triple = u_str + f" {edge_str} " + v_str
            triples.append(remove_extra_spaces(pos_triple))

    return triples


def process_path_string(g, path):
    edges = list(zip(path[:-1], path[1:]))
    triples = get_triples_from_edges(g, edges)
    
    return remove_extra_spaces(f" {SEP} ".join(triples))


def get_triples_from_node(g, n, distance=1):
    triples = list()
    use_stereotype = g.nodes[n]['use_stereotype'] if 'use_stereotype' in g.nodes[n] else False
    g.nodes[n]['use_stereotype'] = False
    node_neighbours = get_node_neighbours(g, n, distance)
    for neighbour in node_neighbours:
        paths = [p for p in nx.all_simple_paths(g, n, neighbour, cutoff=distance)]
        for path in paths:
            triples.append(process_path_string(g, path))
    
    g.nodes[n]['use_stereotype'] = use_stereotype
    return triples


def get_node_str(g, n, distance=1):
    node_triples = get_triples_from_node(g, n, distance)
    return remove_extra_spaces(f" | ".join(node_triples))


def create_triples_from_graph_edges(graphs):
    triples = list()
    for g, _ in tqdm(graphs):
        triples += get_triples_from_edges(g)

    return triples


def mask_graph(graph, stereotypes_classes, mask_prob=0.2, use_stereotypes=False, use_rel_stereotypes=False):
    all_stereotype_nodes = [node for node in graph.nodes if 'stereotype' in graph.nodes[node]\
         and graph.nodes[node]['stereotype'] in stereotypes_classes and has_neighbours_incl_incoming(graph, node)\
            and (True if use_rel_stereotypes else graph.nodes[node]['type'] == 'Class')]
    
    assert all(['stereotype' in graph.nodes[node] for node in all_stereotype_nodes]), "All stereotype nodes should have stereotype property"

    total_masked_nodes = int(len(all_stereotype_nodes) * mask_prob)
    masked_nodes = random.sample(all_stereotype_nodes, total_masked_nodes)
    unmasked_nodes = [node for node in all_stereotype_nodes if node not in masked_nodes]

    for node in masked_nodes:
        graph.nodes[node]['masked'] = True
        graph.nodes[node]['use_stereotype'] = False
    
    for node in unmasked_nodes:
        graph.nodes[node]['masked'] = False
        graph.nodes[node]['use_stereotype'] = use_stereotypes

    assert all(['masked' in graph.nodes[node] for node in all_stereotype_nodes]), "All stereotype nodes should be masked or unmasked"
    
    

def mask_graphs(graphs, stereotypes_classes, mask_prob=0.2, use_stereotypes=False, use_rel_stereotypes=False):
    masked, unmasked, total = 0, 0, 0
    # for graph, f_name in tqdm(graphs, desc='Masking graphs'):
    for graph, _ in graphs:
        mask_graph(graph, stereotypes_classes, mask_prob=mask_prob, use_stereotypes=use_stereotypes, use_rel_stereotypes=use_rel_stereotypes)
        masked += len([node for node in graph.nodes if 'masked' in graph.nodes[node] and graph.nodes[node]['masked']])
        unmasked += len([node for node in graph.nodes if 'masked' in graph.nodes[node] and not graph.nodes[node]['masked']])
        total += len([node for node in graph.nodes if 'masked' in graph.nodes[node]])
        
    ## % of masked nodes upto 2 decimal places
    print(f"Masked {round(masked/total, 2)*100}%")
    print(f"Unmasked {round(unmasked/total, 2)*100}%")

    print("Total masked nodes:", masked)
    print("Total unmasked nodes:", unmasked)