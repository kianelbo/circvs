import math

import matplotlib.pyplot as plt
import networkx as nx

from circvs.datatypes import *


def _closest_factors(n):
    root = int(math.sqrt(n))
    for i in range(root, 0, -1):
        if n % i == 0:
            return i, n // i

def visualize_circuit(edges, n_layers, n_heads, n_positions, token_labels, placement=None):
    head_grid_h, head_grid_w = _closest_factors(n_heads)
    font_size = 6 if n_positions else 14
    if n_positions:
        size_map = {
            Component.EMB: 500,
            Component.LMH: 500,
            Component.MLP: 300,
            Component.SA: 80,
        }
    else:
        size_map = {
            Component.EMB: 1000,
            Component.LMH: 1000,
            Component.MLP: 600,
            Component.SA: 600,
        }

    def place_node_no_positions(node):
        if node.cmpt == Component.EMB:
            return n_heads / 2, node.layer - .5
        if node.cmpt == Component.LMH:
            return n_heads / 2, node.layer + 1
        if node.cmpt == Component.MLP:
            return n_heads, node.layer
        # if node.cmpt == Component.SA:
        return node.head, node.layer

    def place_node_all_positions(node):
        if node.cmpt == Component.EMB:
            return node.position, node.layer - .8
        if node.cmpt == Component.LMH:
            return node.position, node.layer + .8
        if node.cmpt == Component.MLP:
            return node.position, node.layer + 0.3
        # if node.cmpt == Component.SA:
        return node.position + node.head_idx % head_grid_w * .16 - .2, node.layer + node.head_idx // head_grid_h * .16 - .4


    involved_nodes = set()
    for edge in edges:
        involved_nodes.add(edge.src)
        involved_nodes.add(edge.dst)

    if n_positions:
        nodes_list = [Node(Component.MLP, layer, None, pos) for pos in range(n_positions) for layer in range(n_layers)]
        nodes_list += [Node(Component.SA, layer, head, pos) for pos in range(n_positions) for head in range(n_heads) for layer in range(n_layers)]
        nodes_list += [Node(Component.EMB, 0, None, pos) for pos in range(n_positions)]
        nodes_list += [Node(Component.LMH, n_layers - 1, None, pos) for pos in range(n_positions)]
    else:
        nodes_list = [Node(Component.MLP, layer, None, None) for layer in range(n_layers)]
        nodes_list += [Node(Component.SA, layer, head, None) for head in range(n_heads) for layer in range(n_layers)]
        nodes_list.append(Node(Component.EMB, 0, None, None))
        nodes_list.append(Node(Component.LMH, n_layers - 1, None, None))

    uninvolved_nodes = set(nodes_list) - involved_nodes

    graph = nx.DiGraph({n:[] for n in nodes_list})

    for edge in edges:
        graph.add_edge(edge.src, edge.dst, weight=edge.weight)

    if placement is None:
        place_fn = place_node_all_positions if n_positions else place_node_no_positions
        placement = {n: place_fn(n) for n in graph.nodes}

    fig, ax = plt.subplots(figsize=(n_positions, n_layers + 1))
    nx.draw_networkx_nodes(graph, placement, node_size=size_map[Component.LMH], node_shape="s", node_color='w', edgecolors='k', nodelist=[n for n in involved_nodes if n.cmpt == Component.LMH], ax=ax).set_zorder(0)
    nx.draw_networkx_nodes(graph, placement, node_size=size_map[Component.EMB], node_shape="s", node_color='w', edgecolors='k', nodelist=[n for n in involved_nodes if n.cmpt == Component.EMB], ax=ax).set_zorder(0)
    nx.draw_networkx_nodes(graph, placement, node_size=size_map[Component.SA], node_shape="o", node_color='w', edgecolors='k', nodelist=[n for n in involved_nodes if n.cmpt == Component.SA], ax=ax).set_zorder(0)
    nx.draw_networkx_nodes(graph, placement, node_size=size_map[Component.MLP], node_shape="s", node_color='w', edgecolors='k', nodelist=[n for n in involved_nodes if n.cmpt == Component.MLP], ax=ax).set_zorder(0)
    nx.draw_networkx_nodes(graph, placement, node_size=size_map[Component.SA], node_shape="o", node_color='w', edgecolors='gray', nodelist=[n for n in uninvolved_nodes if n.cmpt == Component.SA], ax=ax).set_zorder(0)
    nx.draw_networkx_nodes(graph, placement, node_size=size_map[Component.MLP], node_shape="s", node_color='w', edgecolors='gray', nodelist=[n for n in uninvolved_nodes if n.cmpt == Component.MLP], ax=ax).set_zorder(0)

    nx.draw_networkx_labels(graph, placement, labels={n: n.head_idx if n.head_idx is not None else n.cmpt.name for n in involved_nodes}, font_size=font_size, ax=ax)

    max_w = max(e.weight for e in edges)
    weights = [2 * e.weight / max_w + 1 for e in edges]
    for e, edge in enumerate(graph.edges(data='weight')):
        nx.draw_networkx_edges(graph, placement, edgelist=[edge], width=weights[e], edge_color='r', node_size=50, connectionstyle="arc3, rad=0.1", ax=ax)

    ax.set_ylabel("transformer block layers")
    if n_positions:
        ax.set_xlabel("token positions")
        assert len(token_labels) == n_positions
        ax.set_xticks(range(n_positions), token_labels)
    ax.set_yticks(range(n_layers))
    ax.tick_params(labelleft=True, labelbottom=True)

    if n_positions:
        ax.set_xticks(ax.get_xticks() + 0.5, minor=True)
    ax.set_yticks(ax.get_yticks() + 0.5, minor=True)
    ax.grid(False, which='major')
    ax.grid(True, which='minor', linestyle='--', color='black', alpha=0.75)

    ax.margins(y=0.01, x=0.02)
    fig.tight_layout()
    # plt.grid(False)
    plt.box(False)
