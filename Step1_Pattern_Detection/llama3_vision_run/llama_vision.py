#following: https://github.com/ua-datalab/Generative-AI/blob/main/Notebooks/Using_AI_Verde_OCR.ipynb

import base64
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import torch
import cv2
import PIL.Image
# from pathlib import Path


# General
import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# stuff only required for jupyter display
# from IPython.core.display import Image as DisplayImage
# from IPython.display import display
# from IPython.core.display import HTML



# Setup required parameters to use AI-verde's OpenAI-compatible API
model_llama = "Llama-3.2-11B-Vision-Instruct"
# llm_host = os.environ.get('OPENAI_API_BASE', "https://llm-api.cyverse.org/v1")
llm_host = os.environ.get('OPENAI_API_BASE', "https://llm-api.cyverse.ai")
api_key = os.environ.get('OPENAI_API_KEY', 'your key')

# model_llama.to(device).eval()

# directly using langchain ChatOpenAI
llm = ChatOpenAI(
    model=model_llama,
    temperature=.02,
    api_key=api_key,
    base_url=llm_host,
)

#temperature description: https://stackoverflow.com/questions/79480448/what-is-the-default-value-of-temperature-parameter-in-chatopenai-in-langchain

# print("Test the LLM works")
# print(llm.invoke("Hello, how are you?")) # validate we can talk with the LLM



###########################################################

# https://cloudinary.com/guides/bulk-image-resize/python-image-resize-with-pillow-and-opencv
# resize larger images that are breaking the llm
# attempt to save most of the quality using opencv

def resize(image_file):
    #get and review image
    img = cv2.imread(image_file)
    print('img width is', img.shape[1])
    print('img height is', img.shape[0])

    #resize the image
    new_img = cv2.resize(img, (1000, 762), cv2.INTER_LINEAR)
    print('New img width is', new_img.shape[1])
    print('New img height is', new_img.shape[0])
    new_img_filepath = image_file[:-4]+'_resized.jpg'
    # cv2.imwrite(new_img_filepath, new_img)
    # print(new_img_filepath)
    img_file = new_img_filepath.split('\\')
    # print(img_file[1])
    destination = "blocked_imgs/"+img_file[-1]
    cv2.imwrite(destination, new_img)


    return new_img_filepath

# function to add to JSON
def write_json(new_data, imag_path, json_file):
    try:

        with open(json_file,'r+', encoding='utf-8') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
    except FileNotFoundError:
        file_data = []
    
    img_deetz = {imag_path:{
         new_data
    }
    }
            # Join new_data with file_data inside emp_details
            # file_data["img_details"].append(new_data)
    file_data.append(img_deetz)
            # file_data[imag_path].append(new_data)
            # Sets file's current position at offset.
    # file.seek(0)
            # convert back to json.
    with open(json_file, mode='w', encoding='utf-8') as file:
        json.dump(file_data, file, indent = 4, default = list)

    return file_data




def classify_imgs_large(test_images, prompt, title):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    categories = []
    blocked_files = []
    file_names = []
    image_num = 0
    image_id = []
    for filename in tqdm(test_images):
        # print("filename", filename)
        image_num += 1
        file_type = filename.split(".")[-1] #get the filetype i.e. jpg, png, etc
        image_data = ""
        # file = os.path(filename) #get the filepath
        # file = os.path.join(test_images, filename) #get the filepath

        # with PIL.Image.open(filename) as f:
        # with PIL.Image.open(file) as f:
        with open(filename, 'rb') as f:

        # the image data must be base64 encoded -> convert it
        # then decode the converted image
            image_data = base64.b64encode(f.read()).decode()

        with torch.no_grad():
            torch.cuda.empty_cache()
            
            #use image and prompt to query the llm
            try:
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{file_type};base64,{image_data}"},
                        },
                    ],
                )
                # Call the LLM with the images
                result = llm.invoke([message])

                # show the image (in jupyter notebook)
                # print("\n\n----------\n Image Number:", image_num, "Image File Name:", file)
                # display(DisplayImage(filename=file, width=600, unconfined=True))
                # print the LLM's response
                # print(result.content)
                categories.append(result.content)
                image_id.append(str(image_num))
                file_names.append(filename)
        
            except Exception as e:
                #sometimes image information is too big, even after some preprocessing so it needs to be resized
                print("\n\n",filename, "did not work with api, resizing ...")
                resized_file = resize(filename)
                blocked_files.append(resized_file)
                categories.append("resized_image")
                image_id.append(str(image_num))
                file_names.append("blocked_"+filename)  


        write_json(result.content, filename, json_file=title)


    return image_id, file_names, categories, blocked_files
    # return df_categories


def filter_bad_imgs(img_directory):
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
            with PIL.Image.open(filepath) as img:
                img.verify()  # Only checks that image is not corrupted
            image_paths.append(filepath)  # If verify() passes, we save it
        except Exception as e:
            print(f"Skipping bad image: {filename} — {e}")
            bad_images.append(filename)
            continue

    bad_images = pd.DataFrame(bad_images)
    bad_images.to_csv("blocked_files_llama3.csv")
    clean_imgs = pd.DataFrame(image_paths)
    clean_imgs.to_csv('working_image_paths.csv')

    print(f"\nFound: {len(image_paths)} valid image paths")


    return image_paths

def savefiles(image_id, file_names, categories, file_title):

    df = pd.DataFrame(data={"img_num": image_id, "file_name": file_names, "raw_category_output": categories})
    # zip file names and categories together for a json file/dictionary structure


    np.save(file_title+".npy", df)
    # df = pd.DataFrame(categories25N)
    df.to_csv(file_title+".csv", sep=',',index=False)

    # print(df)
    # df_json = df.groupby('file_name')
    # print(df_json)
    df_json = df.to_dict()

    with open(file_title+".json", "w") as f:
            json.dump(df_json, f, indent=2)

#set prompt
# prompt = "From these categories: 'protest', 'digital flyer in Spanish', digital flyer in English', 'small group of people', 'illustration or cartoon', 'solidarity', 'a person or selfie', 'signs or banners', 'statues (landmarks)/buildings'; Which describes the image best? Please respond with one or two of the categories separated by a ';' then provide a description why you chose these categories." 
prompt = "Only reply with a json file structure. Your response should have the typical json structure:" \
"labels:{ Label_1: please choose the most probably label from this list of labels : 'protest', 'digital flyer in Spanish', digital flyer in English', 'small group of people', 'illustration or cartoon', 'solidarity', 'a person or selfie', 'signs or banners', 'statues (landmarks)/buildings' " \
"         Label_2: please choose the second most probably label from this list of labels : 'protest', 'digital flyer in Spanish', digital flyer in English', 'small group of people', 'illustration or cartoon', 'solidarity', 'a person or selfie', 'signs or banners', 'statues (landmarks)/buildings" \
"}" \
"Description: please create a description of the image" \
"Please don't include any leading or follow up text or comments outside of the json file." \
"Please include blank labels or descriptions if you are unable to provide them. Do not stray from the json structure."



# filtered_imgs = "/home/exouser/filtered_images/filtered_images"
filtered_imgs = "/media/volume/filtered_images_v2_v4/filtered_images"
# filtered_imgs = "filtered_images"
# test_img = "C:\\Users\\lwert\\OneDrive - University of Arizona\\Documents\\UofA\\diss_prop\\insta_scraping\\ImageClassification\\LLM_tests\\fotos_p"
filtered_imgs = filter_bad_imgs(filtered_imgs)
image_id, file_names, categories, blocked_files = classify_imgs_large(filtered_imgs, prompt, "llama3_categories.json")
savefiles(image_id, file_names, categories, "llama_labels_descrps_promp")



