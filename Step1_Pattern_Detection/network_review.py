from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import seaborn as sns
from collections import defaultdict
# from fa2 import ForceAtlas2
import umap




import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm


#############################
# Combibe and process the data

#clip labels
image_labels = np.load("max_label_clip.npy", allow_pickle=True)
#get the items in the pickle file
image_labels = image_labels.item()

labels = list(image_labels.values())
#get the labels in the tuple
label_class = [i[0] for i in labels]
# print(type(labelabel_classls))
# label_class = [i[0] for i in labels]
# print('view all labels')
# print(label_class)


#get the probabilities of the tuple
probabilites = [i[1] for i in labels]
#get the dictionary keys
image_paths = list(image_labels.keys())
paths = [i.split("/")[-1] for i in image_paths]


#create the dataframe to combine later
labels_df = pd.DataFrame({
    "image_file": paths,
    "image_labels": label_class,
    "label_probabilites":probabilites
                      })

labels_df.head()

#blip-2 captions
image_captions = np.load("blip2/blip2_captions.npy", allow_pickle=True)
# print(image_captions)
image_captions = image_captions.item()
captions = list(image_captions.values())
print(captions[:5])
# captions = [i[0] for i in captions]
image_names = list(image_captions.keys())
names = [i.split("/")[-1] for i in image_names]

captions_df = pd.DataFrame({
    "image_file": names,
    "captions": captions,
                      })

# print(captions_df.head())
print("Merging DataFrames")
df = pd.merge(labels_df, captions_df, on="image_file", how='inner')
print(df.head())


##################################
## get semantics of captions
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Encoding Captions")
df['embedding'] = model.encode(df['captions'].tolist(), convert_to_numpy=True, normalize_embeddings=True).tolist()
print(df['embedding'][:5])

###########
# start building the network

G = nx.Graph()

# Add nodes with attributes
print("Adding nodes with attributes")
for _, row in tqdm(df.iterrows()):
    G.add_node(row['image_file'], label=row['image_labels'], caption=row['captions'])

# Compute pairwise similarities
embeddings = np.array(df['embedding'].tolist())
sim_matrix = cosine_similarity(embeddings)

# Add edges based on caption similarity threshold
print("adding edges to network")
threshold = 0.7
for i in tqdm(range(len(df))):
    for j in range(i + 1, len(df)):
        sim = sim_matrix[i][j]
        if sim >= threshold:
            G.add_edge(df.at[i, 'image_file'], df.at[j, 'image_file'], weight=sim)



################################
# plotting the spring layout

# plt.figure(figsize=(10, 8))

# # Spring layout for visualizing
# pos = nx.spring_layout(G, seed=42)

# # Get node colors by label (convert to numbers for coloring)
# label_to_color = {label: i for i, label in enumerate(df['image_labels'].unique())}
# node_colors = [label_to_color[G.nodes[node]['label']] for node in G.nodes()]

# # Draw graph
# nx.draw_networkx_nodes(G, pos, node_size=100, cmap='tab10', node_color=node_colors)
# nx.draw_networkx_edges(G, pos, alpha=0.3)
# plt.title("Image Network (Captions + Labels)")
# plt.axis('off')
# plt.show()



# Build a color map for CLIP labels
# label_palette = sns.color_palette("tab10", n_colors=len(df['image_labels'].unique()))
# label_to_color = {label: label_palette[i] for i, label in enumerate(df['image_labels'].unique())}

# # Node color by CLIP label
# node_colors = [label_to_color[G.nodes[n]['label']] for n in G.nodes]

# # Layout and plot
# pos = nx.spring_layout(G, seed=42)
# plt.figure(figsize=(12, 10))
# nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
# nx.draw_networkx_edges(G, pos, alpha=0.2)
# plt.title("Network of Images Colored by CLIP Labels")
# plt.axis('off')
# plt.show()

#########################
#Identify edges in and outside of subgraphs

# Group nodes by CLIP label
label_groups = defaultdict(list)
for node, data in G.nodes(data=True):
    label_groups[data['label']].append(node)

# Count internal vs external edges
print("Counting internal vs external edges")
for label, nodes in tqdm(label_groups.items()):
    internal = G.subgraph(nodes).number_of_edges()
    external = sum(1 for node in nodes for neighbor in G.neighbors(node) if G.nodes[neighbor]['label'] != label)
    print(f"{label} → internal edges: {internal}, external edges: {external}")



##################
# exploring convex hulls

# print("Exploring convex hulls")
# label_groups_list = defaultdict(list)
# for node in G.nodes():
#     label = G.nodes[node]['label']
#     label_groups_list[label].append(node)

# # Get positions as array for plotting
# pos_array = {k: np.array(v) for k, v in pos.items()}

# colors = plt.cm.tab10.colors
# plt.figure(figsize=(12, 10))

# # Draw hulls
# for i, (label, nodes) in enumerate(label_groups_list.items()):
#     pts = np.array([pos[n] for n in nodes])
#     if len(pts) >= 3:
#         hull = ConvexHull(pts)
#         hull_pts = pts[hull.vertices]
#         plt.fill(hull_pts[:,0], hull_pts[:,1], alpha=0.2, color=colors[i % len(colors)], label=label)

# # Draw nodes and edges
# nx.draw_networkx_edges(G, pos, alpha=0.1)
# nx.draw_networkx_nodes(G, pos, node_color=[colors[i % len(colors)] for i, label in enumerate(nx.get_node_attributes(G, 'label').values())], node_size=50)

# plt.title("Network of Images with Convex Hulls Around CLIP Labels")
# plt.axis('off')
# plt.legend()
# plt.show()


##########################
# plots for larger amounts of nodes

#UMAP
# Compute cosine similarity of caption embeddings
# similarity = cosine_similarity(embeddings)

# Optionally sparsify it (e.g. keep top-k per row)

# Reduce to 2D
# umap_layout = umap.UMAP(metric='precomputed').fit_transform(1 - similarity)

# Format for NetworkX
# positions = {node: coords for node, coords in zip(df['image_file'], umap_layout)}



## UMAP2
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
coords_2d = reducer.fit_transform(embeddings)

# Format for NetworkX layout
pos = {image_id: coord for image_id, coord in zip(df['image_file'], coords_2d)}

nx.draw_networkx_nodes(G, pos, node_size=50)
nx.draw_networkx_edges(G, pos, alpha=0.1)
plt.title("Network of Images with UMAP of Caption Embeddings")
plt.axis('off')
plt.legend()
plt.show()

### forceatlas2
# forceatlas2 = ForceAtlas2(
#     outboundAttractionDistribution=True,  # Hub repulsion
#     linLogMode=False,
#     adjustSizes=False,
#     edgeWeightInfluence=1.0,
#     jitterTolerance=1.0,
#     barnesHutOptimize=True,
#     barnesHutTheta=1.2,
#     scalingRatio=2.0,
#     gravity=1.0,
#     verbose=True
# )

# # G is your NetworkX graph
# positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)