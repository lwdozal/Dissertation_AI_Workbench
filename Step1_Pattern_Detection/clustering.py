'''
Clustering for embeddings
- PCA, k-mediods, and T-SNE
- UMAP and HDBSCAN

#support
# https://umap-learn.readthedocs.io/en/latest/clustering.html
# https://stats.stackexchange.com/questions/263539/clustering-on-the-output-of-t-sne
# https://stackoverflow.com/questions/68398233/clustering-with-umap-and-hdbscan
# https://medium.com/@ps.deeplearning.training/hdbscan-clustering-and-umap-visualisation-f31e653f7218
# https://dylancastillo.co/posts/clustering-documents-with-openai-langchain-hdbscan.html

'''


import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import json

import hdbscan
from umap import UMAP
import collections
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, xlabel, ylabel, show, figure
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from random import randint




#############################################
# data prep


def get_dic(features):
    image_features = features.item()

    feature_embeds = np.array(list(image_features.values()))
    image_path = list(image_features.keys())

    return feature_embeds, image_path


#pad uneven data
def pad_Data(image_features):
    #make sure all vectors are the same size
    target_length = 32
    feature_dim = 2048
    # feature_dim = (512,)
    # add padding for those with a smaller lenth than 32
    padded_features = []
    image_names = []

   
    # for name, feature in image_features.items():
    for name, feature in image_features.item():
        feature = np.array(feature)
        # print(name)

    # for name, feature in enumerate(image_features):


        # Check if it's 2D
        if feature.ndim != 2 or feature.shape[1] != feature_dim:
            print(f"Skipping {name}: unexpected shape {feature.shape}")
            continue

        if feature.shape[0] < target_length:
            # Pad with zeros at the end
            pad_len = target_length - feature.shape[0]
            pad = np.zeros((pad_len, feature_dim))
            feature_padded = np.vstack((feature, pad))
        elif feature.shape[0] > target_length:
            # Truncate to target length
            feature_padded = feature[:target_length, :]
        else:
            feature_padded = feature

        padded_features.append(feature_padded)
        image_names.append(name)

    # Now stack into 3D array if needed or flatten to 2D
    padded_array = np.array(padded_features)  # shape: (n_images, 32, 2048)
                            
    # Example: flatten for clustering
    flattened_features = padded_array.reshape(len(padded_array), -1)  # shape: (n_images, 32*2048)
    print("flattened_features shape", flattened_features.shape)

    return padded_features,flattened_features, image_names


#########
#Pool features incase long compute time / clusters not found

# pool features to reduce noise 
def pool_features(padded_features):
    #pool features if necessary
    pooled_features = np.array([f.mean(axis=0) for f in padded_features])  # shape: (n_images, 2048)
    # Normalize
    scaled = StandardScaler().fit_transform(pooled_features)
    #pca for dimensionality reduction
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(scaled)
    plot(pca.explained_variance_, linewidth=2)
    xlabel('Components')
    ylabel('Explained Variaces')
    show()

    return scaled, reduced

# scaled, reduced = pool_features(padded_features)


##############################################################
## Basic Clustering


def exp_var(features):
    '''    
    #Reduce dimensionality
    # use pca to identify reduced features
    # choose the number of components that explain a certain percentage of the variance (e.g., 95%)
    # documentation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    '''
    # pca2 = PCA().fit(features)
    # print("pca2", len(pca2))
    # pca = PCA(n_components= .95)


    feature_embeds, image_path = get_dic(features)


    # pca = PCA()
    pca = PCA(n_components=50) 
    reduced_features = pca.fit_transform(feature_embeds)
    print("reduced_features", len(reduced_features))
   
    '''
    explained_variance_
    The amount of variance explained by each of the selected components. 
    Equal to n_components largest eigenvalues of the covariance matrix of X.
    https://stackoverflow.com/questions/64015371/how-to-calculate-optimum-feature-numbers-in-pca-python
    '''
    plot(pca.explained_variance_, linewidth=2)
    xlabel('Components')
    ylabel('Explained Variaces')
    show()

    return reduced_features


# #Visualization
def tsne_vis(reduced_features):
    '''
    Evaluate the clusters: 
    https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
    '''
    #Clustering
    kmeans = KMeans(n_clusters=7, random_state=42) #10
    clusters = kmeans.fit_predict(reduced_features)
    tsne = TSNE(n_components=3)
    tsne_results = tsne.fit_transform(reduced_features)

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters)
    plt.colorbar()
    plt.title("Image Clusters: T-SNE")
    plt.show()


##############################################################
## non-linear clustering

# standard_embedding = umap.UMAP(random_state=42).fit_transform(mnist.data)
# plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=mnist.target.astype(int), s=0.1, cmap='Spectral');
def cluster_plt(features, plt_title):
    '''
    UMAP is a dimension reduction or manifold learning algorithm that tries to take high dimensional data and learn essentially some latent structure. 
    It's built on topological reasoning, so it tries to understand the topology of the data and then give a lower dimensional representation of that data 
    that hopefully preserves as much of that structure as it can while still compressing it down to lower dimensions.
    https://arize.com/resource/understanding-umap-and-hdbscan/
    '''
  
    image_features = features.item()
    # image_features = features
    feature_embeds = np.array(list(image_features.values()))
    image_path = list(image_features.keys())
    image_path = [i.split("/")[-1] for i in image_path]


    # feature_embeds = np.array(list(np.array(features)))
    # image_path = list(np.array(path))
    # image_path = [i.split("/")[-1] for i in image_path]


    clusterable_embedding = UMAP(
    n_neighbors=20, #default 10
    min_dist=0.0,
    n_components=2,
    random_state=42,
    # ).fit_transform(features)
    ).fit_transform(feature_embeds)
    # ).fit_transform(feature_embeds, init='random')


    clusterer = hdbscan.HDBSCAN(
        min_samples= 1, #default none
        min_cluster_size=3, #default 5
        )

    labels = clusterer.fit_predict(clusterable_embedding) #works

    clustered = (labels >= 0)
    random_color="#" + f"{randint(0, 0xFFFFFF):06x}"



    figure(figsize=(10, 10), dpi=80)
    plt.scatter(clusterable_embedding[~clustered, 0],
                clusterable_embedding[~clustered, 1],
                c=random_color, #this is what shows the different clusters
                # c='black', #set known color so you know what to compare it to
                s=10,
                # alpha=0.5,
                cmap='Spectral')


    plt.scatter(clusterable_embedding[clustered, 0],
                clusterable_embedding[clustered, 1],
                c=(labels[clustered]),
                s=20,
                cmap='Spectral')
    plt.title(plt_title)
    plt.show()    

    return image_path, labels, clusterer

    ###################
    # for dissertation: DBCV
    # https://medium.com/data-science/tuning-with-hdbscan-149865ac2970
    ###################

def cluster_dict(image_path, labels, clusterer, point_title, summary_path):
    
    cluster_dict = pd.DataFrame({
        "image_path": image_path,
        "cluster_label": labels,
        "probability": clusterer.probabilities_,
        "outlier_score": clusterer.outlier_scores_,
        # "persistence_score": clusterer.cluster_persistence_
        # "unique_labels" : np.unique(labels)
    })

    clusters_df = pd.DataFrame(cluster_dict)
    # clusters_df = pd.DataFrame.from_dict(cluster_dict, orient='index')
    clusters_df.to_csv(point_title, header=True)
    # clusters_df.style

    # df.to_csv(point_title, index = False)
    print("Sample point probabilites")
    print(clusters_df.head(20).to_string(index=False))

    df_clusters_only = clusters_df[clusters_df['cluster_label'] != -1]
    # clusters_only = labels != -1

    # Group by cluster and compute summary statistics
    # cluster_summary = df_clusters_only.groupby('cluster_label').agg(
    cluster_summary = df_clusters_only.groupby('cluster_label').agg(
        cluster_size=('image_path', 'count'),
        avg_probability=('probability', 'mean'),
        avg_outlier_score=('outlier_score', 'mean'),
        # cluster_images=('image_path', image_path),
        # max_persistence_score =clusterer.cluster_persistence_
    ).reset_index()

    # Sort by cluster size or persistence
    cluster_summary = cluster_summary.sort_values(by='cluster_size', ascending=False)

    # Display the summary
    print("Cluster Quality Summary:\n")
    print(cluster_summary.head(20).to_string(index=False))

    # Optionally save to CSV
    cluster_summary.to_csv(summary_path, index=False)
    print(f"Cluster summary saved to: {summary_path}")

   
    return labels, clusterer, clusters_df, cluster_summary




def main():

    print("Reading in file")
   
    ######################################
    # # ResNet50

    # Get Embeddings
    # Load feature dictionary from pickle file
    # with open("ResNet50/features.pkl", "rb") as f:
    #     image_features = pickle.load(f)
    # with open("clip/clip_embeddings.pkl", "rb") as f:
    #     image_features = pickle.load(f)

    # image_features = np.load("ResNet50/resnet50_embed_imgpath.npy", allow_pickle=True)
   
    # print("padding and flattening ResNet50 featueres")
    # padded_features, flattened_features, image_names = pad_Data(image_features)

    # print("K-Medoids and t-sne")
    # reduced_features = exp_var(flattened_features)
    # tsne_vis(reduced_features)

    # print("Running ResNet HDBSCAN and UMAP")
    # plt_title = 'HDBSCAN Clustering of ResNet50 Embeddings\nUMAP Dimensionality Reduction'
    # summary_path = 'cluster_quality_summary_ResNet50.csv'
    # point_title = "point_dataframe_ResNet50.csv"
    # image_path, labels, clusterer = cluster_plt(image_features, plt_title)
    # labels, clusterer, clusters_df, cluster_summary = cluster_dict(image_path, labels, clusterer, point_title, summary_path)



 ######################################
 # # CLIP   
    # # image_features_clip = np.load("clip/v2/clip_embed_imgpath.npy", allow_pickle=True)
    # # image_features_clip = np.load("clip/v3/clip_labels_and_embeddings.json", allow_pickle=True)
    # with open("clip/v3/clip_labels_and_embeddings.json", 'r') as file:
    #     image_features_clip = json.load(file)
    # # print(image_features_clip)
    # image_features_clip = pd.DataFrame(image_features_clip)
    # image_path = image_features_clip["image_path"]
    # print(image_path[:5])
    # embedding = image_features_clip["img_embedding"]
    # print(embedding[:5])

    # image_features_clip = dict(zip(image_path, embedding))

    # print("padding and flattening featueres")
    # padded_features, flattened_features, image_names = pad_Data(image_features_clip)

    # print("K-Medoids and t-sne")
    # reduced_features = exp_var(flattened_features)
    # tsne_vis(reduced_features)
    
    # print("Running CLIP HDBSCAN and UMAP")
    # plt_title = 'HDBSCAN Clustering of CLIP Embeddings\nUMAP Dimensionality Reduction'
    # summary_path = 'cluster_quality_summary_clip_v4.csv'
    # point_title = "point_dataframe_clip_v4.csv"
    # image_path, labels, clusterer = cluster_plt(image_features_clip, plt_title)
    # labels, clusterer, clusters_df, cluster_summary = cluster_dict(image_path, labels, clusterer, point_title, summary_path)


 ######################################
 # # BLIP-2  

    image_features_blip2 = np.load("blip2/blip2_embed_imgpath.npy", allow_pickle=True)

    # print("padding and flattening featueres")
    # padded_features, flattened_features, image_names = pad_Data(image_features_blip2)

    # print("K-Medoids and t-sne")
    # reduced_features = exp_var(flattened_features)
    # tsne_vis(reduced_features)

    # image_features_blip2 = np.load("blip2/blip2_embeddings (1).npy", allow_pickle=True)
    print("Running blip2 HDBSCAN and UMAP")
    plt_title = 'HDBSCAN Clustering of BLIP-2 Embeddings\nUMAP Dimensionality Reduction'
    summary_path = 'cluster_quality_summary_Blip2_v2.csv'
    point_title = "point_dataframe_Blip2_v2.csv"
    image_path, labels, clusterer = cluster_plt(image_features_blip2, plt_title)
    labels, clusterer, clusters_df, cluster_summary = cluster_dict(image_path, labels, clusterer, point_title, summary_path)




    ######################
    # # print("padding and flattening featueres")
    # padded_features, flattened_features, image_names = pad_Data(image_features_clip)

    # # print("K-Medoids and t-sne")
    # reduced_features = exp_var(flattened_features)
    # tsne_vis(reduced_features)
    

    
    # print("Getting Clustering Metrics")
    # title_metrics = "cluster_eval_metrics_CLIP.csv"
    # print_evals(cluster_features, clusterer, title_metrics)



if __name__ == "__main__":
    main()







# def print_evals(cluster_features, clusterer, title_metrics):

#     '''
#     #Evaluation
#     # Silhouette Score: https://towardsdatascience.com/tuning-with-hdbscan-149865ac2970/
#     # - Silhouette Score and similar indexes like it are inappropriate for measuring density-based techniques!!!
#     # Density Based Clustering Validation or DBCV (https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf)
#     # - Accounts for noise and captures the shape property of clusters via densities not distances

#     # evaluation scores
#     # Good clusters = high persistence (e.g., > 0.8).
#     # Bad points = low probabilities_ or high outlier_scores_.
#     #Noise points = label -1
#     ###
#     '''

#     unique_labels = np.unique(cluster_features)
#     print("Unique cluster labels:", unique_labels)

#     # Cluster persistence: tells you the strength of each cluster
#     persistence_score = clusterer.cluster_persistence_
#     print("persistence_score", persistence_score)  # array of persistence values per cluster

#     # Probability for each point
#     point_probabilities = clusterer.probabilities_
#     print("Probability for each point", point_probabilities)  # array of [0, 1] per point

#     # Points with high cluster probability ... adjust threshold as you see fit
#     high_confidence_indices = np.where(clusterer.probabilities_ > 0.1)[0]
#     # Filter image names based on probabilites
#     high_confidence_images = [cluster_features[i] for i in high_confidence_indices]
#     print("High Confidence Images", high_confidence_images)

#     # Outlier scores (higher means more anomalous)
#     outlier_scores = clusterer.outlier_scores_
#     print(outlier_scores)

#     # Count number of items per cluster
#     counter = collections.Counter(cluster_features)
#     print("items per cluster", counter)




#     # sns.displot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True).set(title='Outlier Scores')


#     dict = {'Clusters': unique_labels, 
#             "PersistenceScore": persistence_score,
#             "PointProbabilities": [point_probabilities],
#             "NodesInCluster" : [counter],
#             "HighConfidenceIndicies":[high_confidence_indices],
#             "HighConfidenceImages" : [high_confidence_images],
#             "OutlierScores" : [outlier_scores]
#     }

#     clusters = pd.DataFrame.from_dict(dict, orient='index')
#     clusters.to_csv(title_metrics, header=True)
#     clusters.style
    



