from collections import deque
import re


remove_extra_spaces = lambda txt: re.sub(r'\s+', ' ', txt.strip())

def find_nodes_within_distance(graph, start_node, distance):
    q, visited = deque(), dict()
    q.append((start_node, 0))
    
    while q:
        n, d = q.popleft()
        if d <= distance:
            visited[n] = d
            neighbours = [neighbor for neighbor in graph.neighbors(n) if neighbor != n and neighbor not in visited]
            for neighbour in neighbours:
                if neighbour not in visited:
                    q.append((neighbour, d + 1))
    
    sorted_list = sorted(visited.items(), key=lambda x: x[1])
    return sorted_list


def get_node_neighbours(graph, start_node, distance):
    neighbours = find_nodes_within_distance(graph, start_node, distance)
    max_distance = max(distance for _, distance in neighbours)
    distance = min(distance, max_distance)
    return [node for node, d in neighbours if d == distance]


def has_neighbours_incl_incoming(graph, node):
    edges = list(graph.edges(node))
    edges += list(graph.in_edges(node))
    return len(edges) != 0
