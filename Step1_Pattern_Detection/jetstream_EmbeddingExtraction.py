# 3 step process

# Model Imports
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
import transformers
from transformers import Blip2Processor, Blip2Model, Blip2ForConditionalGeneration
torch.multiprocessing.set_start_method('spawn')# good solution !!!!
from PIL import Image
# from pathlib import Path


# General
import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm



###########################
## figure out how to do this with batch implmentation

#### Read in Models ####
# ResNet50 (NeurIPS)
# CLIP (NeurIPS)
# BLIP-2 (NeurIPS)
# Llama 3 vision

## ideas
# DiNOv2
# SigLIP (https://github.com/mlfoundations/open_clip) -- uses OpenCLIP



#######################################################################
### Feature Processing


#get image paths

def get_imgs(img_directory):
    #identify image folder and get a list of image paths
    # image_dataset_path = img_directory #"filtered_images/"
    # image_paths = [os.path.join(image_dataset_path, img) for img in os.listdir(image_dataset_path)]

    image_paths = []
    bad_images = []

    for filename in os.listdir(img_directory):
        # if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        #     continue

        filepath = os.path.join(img_directory, filename)

        try:
            with Image.open(filepath) as img:
                img.verify()  # Only checks that image is not corrupted
            image_paths.append(filepath)  # If verify() passes, we save it
        except Exception as e:
            print(f"Skipping bad image: {filename} — {e}")
            bad_images.append(filename)
            continue

    bad_images = pd.DataFrame(bad_images)
    bad_images.to_csv("bad_images.csv")
    print(f"\nFound: {len(image_paths)} valid image paths")


    return image_paths



# make sure your images are standardized (same size, same resolution)
def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension



#### ResNet50

# Extract Features with resnet
# - see if this can be done for all models; like add model as parameter
def extract_features_resnet(image_paths, pkl_title):

    # Load Pretrained ResNet50 Model
    #parts of this model are depreciated, find a newer version
    resnet = models.resnet50(weights="IMAGENET1K_V2") #get most up-to-date weights
    # resnet = models.resnet101() no weights

    # Remove the classification layer
    resnet.fc = torch.nn.Identity()  
    resnet.eval()  # Set to evaluation mode

    # set up to extract features using resnet50
    features = {}
    print("Running Restnet on", len(image_paths), "images")
    for img_path in tqdm(image_paths):

        img = Image.open(img_path).convert("RGB")
        # img.show() #print image to make sure it's the image and not the filepath
        input_tensor = preprocess(img)
        with torch.no_grad():
            feature_vector = resnet(input_tensor).numpy().flatten()
        #save features as values to the dictionary; img_path is the key
        features[img_path] = feature_vector


    # Save feature dictionary to a file
    with open(pkl_title, "wb") as f:
        pickle.dump(features, f)

    # torch.save(features, 'resnt50_features.pt') #new
    # features = torch.load('resnt50_batch_features.pt', weights_only=False)

    return features



##### CLIP
# Support: https://medium.com/@paluchasz/understanding-openais-clip-model-6b52bade3fa3

### save the model when you've done some tuning and more classifications 
# i.e. convert it to a class: https://pytorch.org/tutorials/beginner/saving_loading_models.html 


def get_features_clip(image_paths, class_prompts):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_clip = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model_clip.eval()

    clip_label_probs = {}
    clip_path_embeds2 = []


    print("Running CLIP on", len(image_paths), "images")
    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        text_inputs = processor(text=class_prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            torch.cuda.empty_cache()
            img_outputs = model_clip.get_image_features(**inputs)
            feat = img_outputs / img_outputs.norm(p=2, dim=-1, keepdim=True)

        # with torch.no_grad():
        #     torch.cuda.empty_cache()
            text_feat = model_clip.get_text_features(**text_inputs)
            text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)

            logits = feat @ text_feat.T
            probs = logits.softmax(dim=1).squeeze().cpu().numpy()


            clip_path_embeds2.append({
                'image_path': str(img_path),
                'label_probs': {label: float(prob) for label, prob in zip(class_prompts, probs)},   #label_probs,
                'img_embedding': feat.squeeze().cpu().numpy().tolist(),
                'text_embedding': text_feat.squeeze().cpu().numpy().tolist()
            })

    
    # label_probs = dict(zip(class_prompts, probs))  
        # label_probs = list(zip(class_prompts, probs.tolist()))  
            clip_label_probs[str(img_path)] = {label: float(prob) for label, prob in zip(class_prompts, probs)}


    for k, v in clip_label_probs.items():
        print(k)
        print(v)
        break
    
    #save vector embeddings
    np.save("clip_just_img_embeddings_v2.npy", np.array(feat.squeeze().cpu().numpy().tolist()))

    #save embeddings dataframe
    with open("clip_path_embeddings_v2.json", "w") as f:
        json.dump(clip_path_embeds2, f, indent=2)

    #save highest class probability as json
    with open("clip_label_probs_v2.json", "w") as f:
        json.dump(clip_label_probs, f, indent=2)

    return clip_label_probs, clip_path_embeds2


'''
## Next
BLIP-2 or Qwen-VL
'''



def resize_image(image, max_dim=384):
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h))
    return image

def embeddings_blip2(image_paths, device, processor, model_id):

    model = Blip2Model.from_pretrained(model_id).to(device)
    # model.to(device).eval()
    model.eval()


    embeddings = []
    # filenames = []
    imgpath_embeddings = {}

    print("Running BLIP2 Embedder on", len(image_paths), "images")
    for img_path in tqdm(image_paths):

        image = Image.open(img_path).convert("RGB")
            # image = resize_image(image)cd 

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            torch.cuda.empty_cache()  # clears residual memory

            #getting the feature
            outputs = model.vision_model(pixel_values=inputs["pixel_values"])
                # embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            # print("embedding shape", embedding.shape())
            imgpath_embeddings[str(img_path)] = embedding.tolist()

            # embeddings.append(pooled)
        embeddings.append(embedding)
            # filenames.append(img_path.name)

        # except Exception as e:
        #     print(f"Failed: {img_path} — {e}")


    return embeddings, imgpath_embeddings

def captions_BLIP2(image_paths, processor, device):
    torch.cuda.empty_cache()  # clears residual memory

    # Load BLIP-2 processor and model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "Salesforce/blip2-flan-t5-xl"
    # processor = Blip2Processor.from_pretrained(model_id)
    # model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map="auto").to(device)
    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto", torch_dtype=torch.float16).to("cuda")    
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl",
                                                          device_map={"": 0}, 
                                                          torch_dtype=torch.float16) 

    model.eval()

    captions = {}

    print("Getting BLIP2 Captions for", len(image_paths), "images")
    for img_path in tqdm(image_paths):

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        # inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            # print("generating cptions")
            # output = blip_model.generate(**inputs)
        #get captions from features
            # caption = blip_processor.decode(output[0], skip_special_tokens=True)
            torch.cuda.empty_cache()  # clears residual memory

            generated_ids = model.generate(**inputs, max_new_tokens=75)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        #save captions to dictionary
            captions[img_path] = caption



    return captions


def save_pkl_json(title, file):
    #save as json
    #save as pickle
    try:
        with open(title+".pkl", "wb") as f:
            pickle.dump(str(file), f)
        # json_dict = {k: v.tolist() for k, v in file.items()}
        with open(title+".json", "w") as f:
            json.dump(file, f, indent=2)
        #save as pickle
        with open("/media/volume/boot-vol-step1_pattern_detection"+title+".pkl", "wb") as f:
            pickle.dump(str(file), f)
        with open("/media/volume/boot-vol-step1_pattern_detection"+title+".json", "w") as f:
            json.dump(file, f, indent=2)    

    except Exception as e:
        print(f"Failed: {"/media/volume/boot-vol-step1_pattern_detection"+title+" into volume"} — {e}")





def main():
    
    #get images
    # img_directory = "/home/exouser/filtered_images/filtered_images"
    img_directory = "/media/volume/filtered_images_v2_v4/filtered_images_v2"
    # img_directory = "filtered_images" #for /mnt/data
    image_paths = get_imgs(img_directory)


    # _v2 for "/media/volume/filtered_images_v2_v4/filtered_images_v2"
    # _v4 for "/media/volume/filtered_images_v2_v4/filtered_images_v4"

    ##########################
    # Embeddings
    ##########################
    ##resnet features / embeddings
    ##extract features from images connected to the image path
    pkl_title = "resnet_features_v2.pkl"
    image_features = extract_features_resnet(image_paths, pkl_title)
    print('Extracted Features', image_features)

    # ############## CLIP
    ##get CLIP embeddings and labels (tags)
    print("Getting Clip embeddings and tags")
    possible_classes = ['a photo of a protest', 'an image of solidarity', 'an image of an illustration', 'a photo of a group', 'photo of a sign(s) or banner', 'a selfie', 'an image with text in Spanish', 'an image with text in English', 'a photo of landmarks or buildings']

    get_features_clip(image_paths, possible_classes)

    

    ############## BLIP2
    ##set BLIP2 model things

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Salesforce/blip2-flan-t5-xl"
    processor = Blip2Processor.from_pretrained(model_id)
    
    # #get BLIP embeddings
    print("getting blip2 embeddings")
    blip2_mbeddings, imgpath_embeddings = embeddings_blip2(image_paths, device, processor, model_id)

    np.save("blip2_embeddings_v2.npy", np.stack(blip2_mbeddings))
    np.save("blip2_embed_imgpath_v2.npy", imgpath_embeddings)
    save_pkl_json("blip2_embeddings_v2", imgpath_embeddings)


    # #get BLIP captions
    print("getting blip captions")
    imag_captions = captions_BLIP2(image_paths, processor, device)
    np.save("blip2_captions_v2.npy", imag_captions)
    save_pkl_json("blip2_captions_v2", imag_captions)



if __name__ == "__main__":
    main()
'''
## Next
integrate this into a clustering and graph construction pipeline
'''



