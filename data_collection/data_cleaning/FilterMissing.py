#Cleaning evaluation

import pandas as pd
import numpy as np
import re
import os
import shutil




#############################
    ### filter images ###
#############################

# Load the cleaned content CSV
cleaned_df = pd.read_csv("duplicates_removed.csv", encoding='utf-8')
#make sure the file paths in this csv connect to the correct folder. 
#i.e. put this file within the same directory as the images you are trying to identify


# Filter out rows where the filepath doesn't exist


# Specify the target directory
target_dir = "filtered_images"
# Create the directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)  

missing = []
exitisting = []

# Iterate through the filepaths in the cleaned DataFrame


for filepath in cleaned_df['file_path']:

    filepath = str(filepath)
   
        # Check if the file exists
    if os.path.isfile(filepath):
            # Copy the file to the target directory
        shutil.copy(filepath, target_dir)
        exitisting.append(filepath)
    else:
        missing.append(filepath)
        print(f"File not found: {filepath}")

print(f"\n{len(missing)} files are missing\n")
print(f"All other files have been copied to {target_dir}")

# Filter the cleaned DataFrame to exclude missing filepaths
filtered_df = cleaned_df[cleaned_df['file_path'].isin(exitisting)]
# filtered_df['url'] = urls
print('Length of new dataset', len(filtered_df))

# Save the filtered DataFrame to a new CSV
filtered_df.to_csv("content_cleaned.csv", index=False)

#save the missing file names to local them further
missing = pd.DataFrame(missing, columns=['missing_filepath'])
# missing['url'] = urls
missing.to_csv('missing_files.csv', index = False)
