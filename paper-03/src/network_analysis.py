import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# load the dataset
data = pd.read_csv('data/data.csv', header=None)

# create a graph from the dataset
G = nx.Graph()
for tweet in data[0]:
    words = tweet.split()
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words[i+1:]):
            if word1 != word2:
                if G.has_edge(word1, word2):
                    G[word1][word2]['weight'] += 1
                else:
                    G.add_edge(word1, word2, weight=1)

# remove low-weight edges
threshold = 5
for edge in list(G.edges()):
    if G[edge[0]][edge[1]]['weight'] < threshold:
        G.remove_edge(edge[0], edge[1])

# plot the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.1)
nx.draw_networkx_nodes(G, pos, node_size=100)
nx.draw_networkx_edges(G, pos, width=2)
nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
plt.axis('off')
plt.show()
