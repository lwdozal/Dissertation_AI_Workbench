# Pattern Detection
Identify specific themes and topics within the content of the images. Explore how the presentation of the image and account holder comment might change the content themes.

## Analysis Outline

Get embeddings (Images, comments, AI captions) \\

### Exploratory analysis with image embeddings
- How many clusters? 
- What are the main topics?
- Differences between models

### Identify embeddings similarities

Create Structural graph (content-based knowledge representation) 
- Viz_weights + tag_weights
- Viz_weights + generated captions
- Leidan algorithm; Evaluate

### Human in the Loop
- Evaluation metrics
- Content analysis: Visualize tags; summary statistics of tags, datetime, hashtags, likes


## Implementation
Image embedding analysis without labels or captions
- ResNET
- CLIP
- BLIP-2

### Exploratory Analysis
1) Normalize and Reduce Dimensionality
    - PCA, UMAP, t-SNE -> project onto 2D space to visually inspect clusters
2) Cluster the Embeddings
    - Clustering algorithms for Interpretable groupings
    - HDBSCAN → works for noise; Great for interpretability
    - VIsualize Clusters: Use UMAP/t-SNE plots, color-coded by cluster ID
3) Interpret Clusters
    - Metrics: Silhouette Score; Davies-Bouldin Index
    - Parameter Tuning: DUnn Index, Calunski Harabaz, etc
    - Spot check the clusters
    - Use average embedding per cluster to represent its “center.”
    - Sample representative images per cluster.


### Image embedding analysis with labels and captions
Generate Captions and labels in Structured format:
- BLIP-2 : Caption abiliites
- Llama 3 Vision : Labels + caption
- Qwne 2.5-VL : Labels + caption

Clean post comments i.e. lemmetize, translate emojies, rename hashtags, lowercase sentences
Multlingual sentence transformers
- Huggingface sentence transformer
- LaBASE (Language Agnostic BERT sentence encoder)
- GCN?

... Lead into Step2 to begin creation of Semantic Graph