import matplotlib.pyplot  as plt
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path
import imagesize

#test that you can access and view an image
# path = "./insta_scraping/fotos/25N/"
# img = Image.open(path+"CWsQ-Cxqff__0.jpg")
# img.show()



import pandas as pd
import numpy as np
import re
import os
import shutil

# Remove duplicates
content = pd.read_csv("./insta_scraping/datacollection/content.csv") #use your own content
uniq = content["post_id"].unique()
print("Number of Post Ids", len(uniq))


def find_multi_digits(lst):
    #some ids have multiple image numbers
    #get only one of those images since they're all the same
    result = []
    for string in lst:
        # matches = re.findall(r'\d{1,2,3}', string)
        matches = re.findall(r'\d+', string)
        result.extend(matches)
        # print(result)
    result = sorted(result)
    return result

def filter_group(group, digits):
    #group the image ids by image number
    return group[group['post_no'].str.contains(str(digits), na=False)]

def remove_duplicates(df):
    """
    post_id is the unique id of a post given by Instagram 
    -- we might have downloaded this post multiple times so its post_id will be duplicated
    post_num is the number added to the image name by the bot as it was being collected
    -- This helps identify the duplicates and keep only one version

    Remove duplicates based on post_id groups, keeping only rows that match the maximum
    multi-digit sequence in the 'post_no' column of each group.
    """

    filtered_dfs = []

    # Group by 'post_id'
    post_ids = df.groupby('post_id', as_index=False)

    # Iterate through each unique post_id group
    #single out each post id's group of content
    for unique_post_id, group in post_ids:
        print("_____________\nunique_post_id", unique_post_id) #creates a new index within the post group
        print("group", group) #make sure you're going through the groups

        # Get the 'post_no' column values as a list
        group_ids = group['post_no'].tolist()

        # Extract digit sequences from the group's id
        digis = find_multi_digits(group_ids)
        print("found", digis)
        #identify if there's only one post number
        if len(set(digis)) == 1:
            digit = digis
        else:
            # if there are more than one post numbers for the same image id, keep just one
            digit = digis[-1]
            print("kept", digit)            
            
        # Filter the group using the last digit
        filtered_group = filter_group(group, digit)
        filtered_dfs.append(filtered_group)

    # Concatenate all filtered groups into a single DataFrame
    result_df = pd.concat(filtered_dfs, ignore_index=True)
    return result_df




# create .csv file
cleaned = remove_duplicates(content)
print('cleaned overall', cleaned)
cleaned.to_csv("duplicates_removed.csv", index=False)
# 


#############################
    ### Check all IDs Are Present ###

#match post_ids to content.csv and new content
#############################


def check_all_ids_exist(content, cleaned):
    # Extract unique post IDs from both DataFrames
    original_ids = set(content['post_id'].unique())
    cleaned_ids = set(cleaned['post_id'].unique())
    print("original_ids",original_ids)
    print("cleaned_ids", cleaned_ids)

    # Check if all original IDs are in the cleaned DataFrame
    missing_ids = original_ids - cleaned_ids

    if not missing_ids:
        print("All groups were accounted for in the cleaned version!")
    else:
        print(f"The following groups are missing in the cleaned version: {missing_ids}")

    # Count the number of rows per post_id in both DataFrames
    original_counts = content['post_id'].value_counts()
    cleaned_counts = cleaned['post_id'].value_counts()

    # Compare row counts
    discrepancies = original_counts - cleaned_counts.reindex(original_counts.index, fill_value=0)
    if discrepancies.any():
        print("There are discrepancies in row counts for some groups:")
        print(discrepancies[discrepancies != 0])
    else:
        print("Row counts match for all groups!")

check_all_ids_exist(content, cleaned)