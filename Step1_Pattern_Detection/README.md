# Pattern Detection
Identify specific themes and topics within the content of the images. Explore how the presentation of the image and account holder comment might change the content themes.

## Analysis Outline

Get embeddings (Images, comments, AI captions) \\

### Exploratory analysis with image embeddings
- How many clusters? 
- What are the main topics?
- Differences between models


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
    - K-means, UMAP, -> project onto 2D space to visually inspect clusters
2) Cluster the Embeddings
    - Clustering algorithms for Interpretable groupings
    - HDBSCAN → works for noise; Great for interpretability
    - Vsualize Clusters: Use UMAP/ plots, color-coded by cluster ID
3) Interpret Clusters
    - Metrics: Silhouette Score; Davies-Bouldin Index
    - Parameter Tuning: Calunski Harabaz Index
    - Human in the loop Inductive content analysis: Sample representative images per cluster.


### M-LLM analysis with labels and captions
Generate Captions and labels in Structured format:
-  CLIP : Labels
-  BLIP-2 : Caption abiliites
-  Llama 3 Vision : Labels + Description
-  Qwen 2.5-VL : Labels + Description
-  Llama3: Labels + Description
-  Llama4 Scout: Labels + Description
-  Phi4: Labels + Description


... Lead into Step2 to begin creation of Semantic Graph
