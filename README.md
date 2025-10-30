# AI Workbench: Visual Network Narrative Dissertation
MLOPs and LLMOPs for dissertation analysis workflow

### Implementation

1. Download/clone the repository and save to your desired folder 
2. Create a new virtual environment


## [Data Collection and Evaluation](https://github.com/lwdozal/Dissertation_AI_Workbench/tree/main/data_collection)
- Web Scraping (Bot) 
- Cleaning; Preparing and Monitoring   
- Identify security and ethical risks in the data and storage
### Resources
Instagram Account

## [Step 1: Pattern Detection](https://github.com/lwdozal/Dissertation_AI_Workbench/tree/main/Step1_Pattern_Detection)

### Model Development (Clustering and Network Building)

### Model Development (Image Label and Caption Generation)
Training and fine-tuning on a subset of data to track performance, identify errors, and optimize models.\
Ongoing monitoring of security and ethical risks

### Resources
Hugging Face Access, Access to LLM (I used VERDE), GPU Access

<!-- Torch, Torchvision, \
transformers, sentence transformers,  \
PIL, Requests, pydantic, open-cv, os \
langchain core and openai, \ -->



## Step 2: Pattern Refinement
Clean post comments i.e. lemmetize, translate emojies, rename hashtags, lowercase sentences
Multlingual sentence transformers
- Huggingface sentence transformer
- LaBASE (Language Agnostic BERT sentence encoder)
- GCN?

### Identify embeddings similarities

Create Structural graph (content-based knowledge representation) 
- Viz_weights + tag_weights
- Viz_weights + generated captions
- Leidan algorithm; Evaluate
