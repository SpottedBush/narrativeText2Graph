import hashlib
import networkx as nx
import numpy as np

def stable_hash(x):
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16) % (10**8)

def wl_step(G, labels):
    new_labels = {}
    for n in G.nodes():
        neigh = sorted(labels[v] for v in G.neighbors(n))
        signature = (labels[n], tuple(neigh))
        new_labels[n] = stable_hash(signature)
    return new_labels

def histogram(values, bins=64):
    return np.bincount([v % bins for v in values], minlength=bins)

def embed_graph(G, wl_iterations=2, bins=64):
    labels = {n: stable_hash(G.nodes[n].get("label", 1)) for n in G.nodes()}
    hist_list = []

    # WL iterations
    for _ in range(wl_iterations):
        node_hist = histogram(list(labels.values()), bins=bins)
        hist_list.append(node_hist)
        edge_labels = []
        for u, v in G.edges():
            pair = tuple(sorted((labels[u], labels[v])))
            edge_labels.append(stable_hash(pair))
        edge_hist = histogram(edge_labels, bins=bins)
        hist_list.append(edge_hist)
        labels = wl_step(G, labels)

    n = G.number_of_nodes()
    m = G.number_of_edges()

    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = 0.0

    try:
        diameter = nx.diameter(G) if nx.is_connected(G) else 0.0
    except:
        diameter = 0.0

    global_feats = np.array([n, m, avg_clustering, diameter], dtype=float)

    embedding = np.concatenate(hist_list + [global_feats])
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding
