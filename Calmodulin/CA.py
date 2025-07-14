import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
# Assuming dccm_matrix.dat is in a format where each line is a row of the matrix
dccm_matrix = np.loadtxt('matrix.dat')
num_residues = dccm_matrix.shape[0]  # Number of residues based on matrix dimensions
# Initialize a graph
G = nx.Graph()

# Define a correlation threshold (e.g., |0.5|) for significant connections
threshold = 0.5

# Add nodes and edges
for i in range(num_residues):
    for j in range(i+1, num_residues):  # Only upper triangle to avoid duplicates
        correlation = dccm_matrix[i, j]
        if abs(correlation) >= threshold:
            G.add_edge(i, j, weight=correlation)
# Draw network with nodes and edges weighted by correlation
pos = nx.spring_layout(G)  # Position nodes using a force-directed algorithm
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.title("Network Representation of GeoHNH-AcrIIC1 System")
plt.show()
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
print("Top 5 residues by Betweenness Centrality:", sorted_betweenness[:5])

# Closeness Centrality
closeness_centrality = nx.closeness_centrality(G)
sorted_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
print("Top 5 residues by Closeness Centrality:", sorted_closeness[:5])
# Example: Shortest path between residue 10 and residue 50 (source-sink pair)
source = 10
sink = 50
try:
    shortest_path = nx.shortest_path(G, source=source, target=sink, weight='weight')
    path_length = nx.shortest_path_length(G, source=source, target=sink, weight='weight')
    print(f"Shortest path from residue {source} to residue {sink}: {shortest_path}")
    print(f"Path length: {path_length}")
except nx.NetworkXNoPath:
    print(f"No path exists between residue {source} and residue {sink}")
# Communication efficiency (example)
efficiency = 1 / path_length if path_length > 0 else 0
print(f"Communication efficiency between residue {source} and residue {sink}: {efficiency}")

additional_pairs = [(sorted_betweenness[i][0], sorted_betweenness[j][0]) for i in range(5) for j in range(i + 1, 5)]
for src, tgt in additional_pairs:
    try:
        path = nx.shortest_path(G, source=src, target=tgt, weight='weight')
        path_len = nx.shortest_path_length(G, source=src, target=tgt, weight='weight')
        eff = 1 / path_len if path_len > 0 else 0
        print(f"Path from residue {src} to {tgt}: {path} with efficiency: {eff}")
    except nx.NetworkXNoPath:
        print(f"No path exists between residue {src} and residue {tgt}")
