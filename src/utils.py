import networkx as nx
from matplotlib.patches import Patch
from textwrap import shorten

import matplotlib.pyplot as plt

def visualize_graph(graph,
                    nodes=None,
                    node_type_colors=None,
                    figsize=(12, 8),
                    layout='spring',
                    label_max_width=40,
                    show_legend=True,
                    seed=42):
    """
    Visualize a NetworkX DiGraph with node colors determined by node type.

    - graph: networkx graph (nodes are expected to be node IDs used in `nodes` list or have attributes).
    - nodes: optional list of node dicts (as present in this notebook) containing at least 'id', 'type', 'text'.
             If not provided, the function will try to read attributes from graph.nodes[node].
    - node_type_colors: dict mapping node type -> color (e.g. {'entity': '#1f78b4', 'event': '#ff7f0e'})
    - layout: one of 'spring', 'kamada_kawai', 'circular', 'shell', 'spectral', or 'graphviz' (if pygraphviz/pydot available)
    - label_max_width: truncate node labels to this many characters for readability
    Returns the computed node positions (pos).
    """
    # default colors
    if node_type_colors is None:
        node_type_colors = {
            'entity': '#1f78b4',   # blue
            'event': '#ff7f0e',    # orange
            'segment': '#2ca02c',  # green
        }

    # build lookup from provided nodes list if available
    node_lookup = {}
    if nodes is not None:
        for n in nodes:
            nid = n.get('id') or n.get('node') or n.get('name')  # be tolerant
            if nid is not None:
                node_lookup[nid] = n

    # determine node types and labels
    node_list = list(graph.nodes())
    types = []
    labels = {}
    for nid in node_list:
        ninfo = node_lookup.get(nid, {}) if node_lookup else {}
        # fall back to graph attributes
        gattrs = graph.nodes[nid] if nid in graph.nodes else {}
        ntype = ninfo.get('type') or gattrs.get('type') or gattrs.get('label') or 'unknown'
        types.append(ntype)
        if ntype == 'event':
            text = ''.join(ninfo.get('subject_texts', [])) + ninfo.get('lemma', '[]') + ''.join(ninfo.get('object_texts', []))
        else:
            text = ninfo.get('text') or gattrs.get('text') or str(nid)
        labels[nid] = shorten(text, width=label_max_width, placeholder='…')

    # map types to colors
    color_map = [node_type_colors.get(t, '#BBBBBB') for t in types]

    # compute layout
    if layout == 'spring':
        pos = nx.spring_layout(graph, seed=seed)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(graph)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'shell':
        pos = nx.shell_layout(graph)
    elif layout == 'spectral':
        pos = nx.spectral_layout(graph)
    elif layout == 'graphviz':
        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
        except Exception:
            pos = nx.spring_layout(graph, seed=seed)
    else:
        pos = nx.spring_layout(graph, seed=seed)

    # draw
    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(graph, pos, arrowstyle='-|>', arrowsize=12, edge_color="#777777", alpha=0.8)
    nx.draw_networkx_nodes(graph, pos, node_color=color_map, node_size=800, edgecolors='#333333', linewidths=0.5)
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)

    plt.axis('off')

    # legend
    if show_legend:
        present_types = sorted({t for t in types})
        patches = []
        for t in present_types:
            patches.append(Patch(color=node_type_colors.get(t, '#BBBBBB'), label=t))
        if patches:
            plt.legend(handles=patches, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.show()
    return pos