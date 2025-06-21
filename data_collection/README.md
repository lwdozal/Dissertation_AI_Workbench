# Data Collection and Evaluation
While wihtin your activated environment, install the modules in the requirements. 
`pip install -r requirements.txt` \
Before beginning, make sure you create an Instagram accoutn. I created one jsut for this project.

## Web scraping
Create a list of hashtags you want to use to scrape instagram. \
- In the insta_hashtags2025.py input your password, username, and create a hashtag variable. \
- Uncomment the function calls and url when necessary \
- run insta_accounts2025.py \
- This will open a web browser. Wait for it to sign in using your input password and username. \
- Manually click the 'Not Now' button -- could not figure this out for some reason.

Create a list of accounts you want to use to scrape instagram.
- In the insta_hashtags2025.py input your password, username, and create a hashtag variable. \
- Uncomment the function calls and url when necessary \
- run insta_accounts2025.py \
- This will open a web browser. Wait for it to sign in using your input password and username. \
- Manually click the 'Not Now' button -- could not figure this out for some reason.

Both of these methods will append the output to the *content.csv* file and will create folders for each hashtag/account.

## Cleaning
go to data_ceaning/ folder \
Open the `RemoveDuplicates.py` and make sure the file being as the content vairable matches your file. \
This will create a new file with the duplicates removed from your content file. \

In this new file, *duplicates_removed.csv*, should have the paths to all the images that were downloaded. Some of these images might have downloaded with an error. We want to check if there are any missing from our scraping. \
First:
- Make sure the file paths in this csv connect to the correct folder. i.e. put this file within the same directory as the images you are trying to identify

## Preparing and Monitoring   
-- Identify security and ethical risks in the data and storage



