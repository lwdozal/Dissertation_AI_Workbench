
# import required modules
from selenium import webdriver
import time, urllib.request
# from instascrape import Reel
from bs4 import BeautifulSoup as bs
import selenium.common.exceptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import requests


import pandas as pd
import os


# import PIL  
# from PIL import Image  


#get URl path
def path():
    global driver

    driver = webdriver.Chrome()
	# return driver

# go to page url
def url_name(url):
	# global driver
	# driver = webdriver.Chrome()
	driver.get(url)
	time.sleep(15)

def login(url):
    global driver
    driver = webdriver.Chrome()
    driver.get(url)

    ####### Log in to Instagram
    time.sleep(15) #you're not a robot wait 5 seconds to run the next code script
    username=driver.find_element("css selector","input[name='username']")
    password=driver.find_element("css selector","input[name='password']")
    username.clear() #clear default text in input area
    password.clear() #clear default text in input area
    username.send_keys("USERNAME")
    password.send_keys("PASSWORD")

    login = driver.find_element("css selector","button[type='submit']").click()

# skip pop-ups
    time.sleep(10)
	

# function to get first post
def first_post():
	time.sleep(3)
	#div[class="_aagw"] is the correct version, the second pic variable helps when you've been tesing instagram too many times
	pic = driver.find_element(By.CSS_SELECTOR,'div[class="_aagw"]').click()
	# pic = driver.find_element(By.CSS_SELECTOR,'span[class="html-span xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x1hl2dhg x16tdsg8 x1vvkbs"]').click()
	print("opened first post")
	# print("first_post pic:",pic)
	print("first_post url", driver.current_url)

	time.sleep(2)
	return driver.current_url

# function to get next post
def next_post():
	time.sleep(1.5)
	try:
		nex = driver.find_element(By.CSS_SELECTOR,"div[class=' _aaqg _aaqh']") # _abl-
		return nex
	except selenium.common.exceptions.NoSuchElementException:
		return 0
	

##################################
### Check if things are things ###
##################################

# function to check if the post is nested
def nested_check():

	try:
		time.sleep(1)
		nes_nex = driver.find_element(By.CSS_SELECTOR,"div[class=' _9zm2']")
		# nes_nex = driver.find_element(By.CSS_SELECTOR,"div[class='x1ey2m1c x9f619 xds687c x10l6tqk x17qophe x13vifvy x1ypdohk']")
		print("Nested Post")
		return nes_nex
	
	except selenium.common.exceptions.NoSuchElementException:
		print("Found all posts")
		return 0

# Check if there's a video
def video_check():
	print("checking for video")
	try:
		vid = driver.find_element(By.CSS_SELECTOR, 'div[class = "x5yr21d x1uhb9sk xh8yej3"]')
		return vid
	except selenium.common.exceptions.NoSuchElementException:
		print("Did not find video")
		return None
	
# check to see if the post has likes
def likes_check():
	print("checking for likes")
	try:
		likes = driver.find_element(By.CSS_SELECTOR,"span[class='html-span xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x1hl2dhg x16tdsg8 x1vvkbs']").text
		return likes
	except selenium.common.exceptions.NoSuchElementException:
		print("Did not find likes")
		return None

# check to see if post has comments
def comments_check():
	print("checking for comments")
	try:
		comments = driver.find_element(By.CSS_SELECTOR, 'h1[class = "_ap3a _aaco _aacu _aacx _aad7 _aade"]').text
		return comments
	except selenium.common.exceptions.NoSuchElementException:
		print("Did not find comments")
		return None


##################################################

# download the thumbnail
def download_img(hashtag, img_src, img_call):
	img_Data = requests.get(img_src).content
	with open(hashtag + "01_/" + img_call + ".jpg", 'wb') as handler:
		handler.write(img_Data)

def first_image(hashtag, img_call, c):
	WebDriverWait(driver, 3)
	
	html = driver.page_source
	# print(html)
	soup = bs(html, 'html.parser')
	img_source_first = soup.findAll('div', class_="_aagv")[-2].find('img')['src']
	print("img_source",img_source_first)
	img_alt_first = soup.findAll('div', class_="_aagv")[-2].find('img')['alt']
	jpg_first = urllib.request.urlretrieve(img_source_first, hashtag+'/'+ img_call + 'post_{}firstmby.jpg'.format(str(c)))

########## likes
	if likes_check() != None:
		# likes = driver.find_element(By.CSS_SELECTOR,"span[class='html-span xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x1hl2dhg x16tdsg8 x1vvkbs']").text
		likes = likes_check()
		print("likes",likes)
	else:
		likes = 1
		print("likes",likes)

	######### comments
	if comments_check() != None: 	
		comment = driver.find_element(By.CSS_SELECTOR, 'h1[class = "_ap3a _aaco _aacu _aacx _aad7 _aade"]').text
		print("comment",comment)
	else:
		comment = ""
		print("comment",comment)
	# comment = driver.find_element(By.CSS_SELECTOR, 'h1[class = "_ap3a _aaco _aacu _aacx _aad7 _aade"]').text
	# print("Comment:", comment)

	# account = driver.find_element(By.CSS_SELECTOR, "a[class='x1i10hfl xjqpnuy xa49m3k xqeqjp1 x2hbi6w xdl72j9 x2lah0s xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r x2lwn1j xeuugli x1hl2dhg xggy1nq x1ja2u2z x1t137rt x1q0g3np x1lku1pv x1a2a7pz x6s0dn4 xjyslct x1ejq31n xd10rxx x1sy0etr x17r0tee x9f619 x1ypdohk x1f6kntn xwhw2v2 xl56j7k x17ydfre x2b8uid xlyipyv x87ps6o x14atkfc xcdnw81 x1i0vuye xjbqb8w xm3z3ea x1x8b98j x131883w x16mih1h x972fbf xcfux6l x1qhh985 xm0m39n xt0psk2 xt7dq6l xexx8yu x4uap5 x18d9i69 xkhd6sd x1n2onr6 x1n5bzlp xqnirrm xj34u2y x568u83']").text		
	# account = driver.find_element(By.CSS_SELECTOR, "a[class='x1i10hfl xjbqb8w x1ejq31n xd10rxx x1sy0etr x17r0tee x972fbf xcfux6l x1qhh985 xm0m39n x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz  _acan _acao _acat _acaw _aj1- _ap30 _a6hd']").text
	# print("account",account)
	time = driver.find_element(By.CSS_SELECTOR,"time[class='x1p4m5qa']").get_attribute("datetime")
	# time = driver.find_element(By.CSS_SELECTOR,"time[class='_a9ze _a9zf']").get_attribute("datetime")
	print("DateTime:", time)

    #save everything to csv
	s1 = pd.Series(driver.current_url, name='URL')
	# s2 = pd.Series(hashtag + img_call + '_post_'+str(c), name='img_name')
	s2 = pd.Series(img_call, name='img_name')
	s3 = pd.Series(img_source_first, name='img_src')
	s4 = pd.Series(img_alt_first, name='img__alt')
	s5 = pd.Series(hashtag, name='accounts')
	s6 = pd.Series(likes, name='like_count')
	s7 = pd.Series(comment, name='initial_comment')
	s8 = pd.Series(time, name='datetime')
	s9 = pd.Series(jpg_first[0], name='post_file')
	s10 = pd.Series("", name='video_src')
	s11 = pd.Series(hashtag + '/'+ img_call + 'post_{}firstmby.jpg'.format(str(c)), name='file_path')
	# s12 = pd.Series("post_"+c+"_00", name = "post_id")
	s13 = pd.Series(img_call, name = "post_num")
	df = pd.concat([s1,s2, s3,s4,s5,s6,s7,s8,s9,s10, s11, s13], axis=1)
	#append df to existing csv (create one before running)
	df.to_csv('content.csv', mode='a', header=False)
	print("First Post Data appended successfully.")
	# return img_source_first, img_alt_first


def img_src_alt(hashtag, img_cnt, img_call):
	WebDriverWait(driver, 3)
	print('located image')

	html = driver.page_source
	# print(html)
	soup = bs(html, 'html.parser')
	
    #use
	img_source = soup.findAll('div', class_="_aagv")[-1].find('img')['src']
	print("img_source",img_source)

	WebDriverWait(driver, 3)
	# img_source_alt = soup.findAll('div', class_="_aagv")[-1].find('img')['alt']
	img_source_alt = soup.findAll('div', class_="_aagv")[-1].find('img')
	print("image_alt", img_source_alt)
	# download_img(hashtag, img_source, img_call)
	jpg = urllib.request.urlretrieve(img_source, hashtag + '01/' + img_call + '_{}.jpg'.format(img_cnt))

	if likes_check() != None:
		likes = driver.find_element(By.CSS_SELECTOR,"span[class='html-span xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x1hl2dhg x16tdsg8 x1vvkbs']").text
		print("likes",likes)
	else:
		likes = 1
		print("likes",likes)
		# tags = driver.find_elements(By.CSS_SELECTOR, 'h1[class="x1i10hfl xjbqb8w x1ejq31n xd10rxx x1sy0etr x17r0tee x972fbf xcfux6l x1qhh985 xm0m39n x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz notranslate _a6hd]')
		
######### comments
	if comments_check() != None: 	
		comment = driver.find_element(By.CSS_SELECTOR, 'h1[class = "_ap3a _aaco _aacu _aacx _aad7 _aade"]').text
		print("comment",comment)
	else:
		comment = ""
		print("comment",comment)
		# tags = driver.find_elements(By.CSS_SELECTOR, 'h1[class="x1i10hfl xjbqb8w x1ejq31n xd10rxx x1sy0etr x17r0tee x972fbf xcfux6l x1qhh985 xm0m39n x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz notranslate _a6hd]')
	
	account = hashtag
	# account = driver.find_element(By.CSS_SELECTOR, "a[class='x1i10hfl xjqpnuy xa49m3k xqeqjp1 x2hbi6w xdl72j9 x2lah0s xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r x2lwn1j xeuugli x1hl2dhg xggy1nq x1ja2u2z x1t137rt x1q0g3np x1lku1pv x1a2a7pz x6s0dn4 xjyslct x1ejq31n xd10rxx x1sy0etr x17r0tee x9f619 x1ypdohk x1f6kntn xwhw2v2 xl56j7k x17ydfre x2b8uid xlyipyv x87ps6o x14atkfc xcdnw81 x1i0vuye xjbqb8w xm3z3ea x1x8b98j x131883w x16mih1h x972fbf xcfux6l x1qhh985 xm0m39n xt0psk2 xt7dq6l xexx8yu x4uap5 x18d9i69 xkhd6sd x1n2onr6 x1n5bzlp xqnirrm xj34u2y x568u83']").text		
	# account = driver.find_element(By.CSS_SELECTOR, "a[class='x1i10hfl xjbqb8w x1ejq31n xd10rxx x1sy0etr x17r0tee x972fbf xcfux6l x1qhh985 xm0m39n x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz  _acan _acao _acat _acaw _aj1- _ap30 _a6hd']").text
	# print("account",account)
	time = driver.find_element(By.CSS_SELECTOR,"time[class='x1p4m5qa']").get_attribute("datetime")
	# time = driver.find_element(By.CSS_SELECTOR,"time[class='_a9ze _a9zf']").get_attribute("datetime")
	print("DateTime:", time)

	return (img_source, img_source_alt, account, likes, comment, time)
	# return (img_srcs, img_alts, account, likes, comment, time)


def vid_src_alt(hashtag, img_cnt, img_call):
	WebDriverWait(driver, 3)

	# vid_src = driver.find_element(By.CSS_SELECTOR, 'div[class = "x5yr21d x1uhb9sk xh8yej3"]').get_attribute('src')
	vid_src = driver.find_element(By.CSS_SELECTOR, 'video[class= "x1lliihq x5yr21d xh8yej3"]').get_attribute('src')
	print("vid_src", vid_src)
	thumbnail_alt = "video, no automated alt"
	print("thumbnail_alt", thumbnail_alt)
	
	if likes_check() != None:
		likes = driver.find_element(By.CSS_SELECTOR,"span[class='html-span xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x1hl2dhg x16tdsg8 x1vvkbs']").text
		print("likes",likes)
	else:
		likes = 1
		print("likes",likes)
		# tags = driver.find_elements(By.CSS_SELECTOR, 'h1[class="x1i10hfl xjbqb8w x1ejq31n xd10rxx x1sy0etr x17r0tee x972fbf xcfux6l x1qhh985 xm0m39n x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz notranslate _a6hd]')
	
	######### comments
	if comments_check() != None: 	
		comment = driver.find_element(By.CSS_SELECTOR, 'h1[class = "_ap3a _aaco _aacu _aacx _aad7 _aade"]').text
		print("comment",comment)
	else:
		comment = ""
		print("comment",comment)

	account = hashtag

	time = driver.find_element(By.CSS_SELECTOR,"time[class='x1p4m5qa']").get_attribute("datetime")
	# time = driver.find_element(By.CSS_SELECTOR,"time[class='_a9ze _a9zf']").get_attribute("datetime")
	print("DateTime:", time)
	
	return (vid_src, thumbnail_alt, account, likes, comment, time)


def save_carousel_post(hashtag, img_name):
	WebDriverWait(driver, 3)

	img_url = driver.current_url
	img_call = img_url.split('/')[-2]
	print("img_call", img_call)

	img_src = []
	img__alt = []
	accounts = []
	post_like = []
	comments = []
	datetime = []
	img_file = []
	video_src = []
	img_cnt = 0


	print("found carousel")
	if video_check() != None:
	# if driver.find_element(By.CSS_SELECTOR, 'div[class = "x5yr21d x1uhb9sk xh8yej3"]') is not None:	
		print("a video starts off the carousel")
		vid_src, thumbnl_alt, account, likes, comment, time = vid_src_alt(hashtag, img_cnt, img_call)
		accounts.append(account)
		post_like.append(likes)
		comments.append(comment)
		datetime.append(time)
		# vid_src = vid_src.split("blob:")[-1]
		# video_src.append(vid_src)
		video_src.append(img_name +'_{}.jpg'.format(img_call))
		img_src.append(vid_src)
		img__alt.append(thumbnl_alt)
		# jpg = urllib.request.urlretrieve(vid_src, img_name +'_{}.jpg'.format(img_call))
		# img_file.append(jpg[0])
		img_cnt +=1
		print("image count", img_cnt)

	# 		# nested_check().click()        
		driver.find_element(By.CSS_SELECTOR,'button[class = "_afxw _al46 _al47"], div[class=" _9zm2"]').click()
		print('Video first: clicked on next carousel iamge')

	elif video_check() == None:
		print("an image starts off the carousel")
		src, alt, account, likes, comment, time = img_src_alt(hashtag, img_cnt, img_call)
		# img_src.append(first_img)
		# img_src.append(first_alt)
		img_src.append(src)
		img__alt.append(alt)
		accounts.append(account)
		post_like.append(likes)
		comments.append(comment)
		datetime.append(time)
		# img_cnt +=1
		print("image count", img_cnt)
		# for i in src:
		jpg = urllib.request.urlretrieve(src, img_name + '_{}.jpg'.format(img_call))
		print("jpg path", jpg[0])
		img_file.append(jpg[0])
		img_cnt +=1

	# 		# nested_check().click()        
		driver.find_element(By.CSS_SELECTOR,'button[class = "_afxw _al46 _al47"], div[class=" _9zm2"]').click()

	else:
		print("did not find anything in post")
		
    #save everything to csv
	s1 = pd.Series(driver.current_url, name='URL')
	s2 = pd.Series(img_name, name='img_name')
	s3 = pd.Series(img_src, name='img_src')
	s4 = pd.Series(img__alt, name='img__alt')
	s5 = pd.Series(accounts, name='accounts')
	s6 = pd.Series(post_like, name='like_count')
	s7 = pd.Series(comments, name='initial_comment')
	s8 = pd.Series(datetime, name='datetime')
	s9 = pd.Series(img_file, name='post_file')
	s10 = pd.Series(video_src, name='video_src')
	s11 = pd.Series(img_name + img_call + ".jpg", name='file_path')
	# s12 = pd.Series(img_name, name = "post_id")
	s13 = pd.Series(img_call, name = "post_num")
	df = pd.concat([s1,s2, s3,s4,s5,s6,s7,s8,s9,s10, s11, s13], axis=1)

	#append df to existing csv (create one before running)
	df.to_csv('content.csv', mode='a', header=False)
	print("Data appended successfully.")
	

def save_sngl_content_post(hashtag, img_name):
	WebDriverWait(driver, 3)

	img_url = driver.current_url
	img_call = img_url.split('/')[-2]
	print("img_call - single", img_call)

	img_src = []
	img__alt = []
	accounts = []
	post_like = []
	comments = []
	datetime = []
	img_file = []
	video_src = []
	img_cnt = 0
	
	if video_check() is not None:
		print("getting single video post and metadata")
		vid_src, thumbnail_src, account, likes, comment, time = vid_src_alt(hashtag, img_cnt, img_call)
		accounts.append(account)
		post_like.append(likes)
		comments.append(comment)
		datetime.append(time)
		img_src.append(thumbnail_src)
		# vid_src = vid_src.split("blob:")[-1]
		video_src.append(vid_src)
		# jpg = urllib.request.urlretrieve(vid_src, img_name + '{}.jpg'.format(img_call))
		# print("mp4 path", jpg)
		# img_file.append(jpg[0])
		img_cnt += 1

	else:
		print("getting single image post and metadata")
		src1, alt1, account, likes, comment, time = img_src_alt(hashtag, img_cnt, img_call)
		img_src.append(src1)
		img__alt.append(alt1)
		accounts.append(account)
		post_like.append(likes)
		comments.append(comment)
		datetime.append(time)
		jpg = urllib.request.urlretrieve(src1, img_name + '{}.jpg'.format(img_call))
		print("jpg path", jpg[0])
		img_file.append(jpg[0])
		img_cnt +=1

    #save everything to csv
	s1 = pd.Series(driver.current_url, name='URL')
	s2 = pd.Series(img_name, name='img_name')
	s3 = pd.Series(img_src, name='img_src')
	s4 = pd.Series(img__alt, name='img__alt')
	s5 = pd.Series(accounts, name='accounts')
	s6 = pd.Series(post_like, name='like_count')
	s7 = pd.Series(comments, name='initial_comment')
	s8 = pd.Series(datetime, name='datetime')
	s9 = pd.Series(img_file, name='post_file')
	s10 = pd.Series(video_src, name='video_src')
	s11 = pd.Series(img_name + img_call + ".jpg", name='file_path')
	# s12 = pd.Series(img_name, name = "post_id")
	s13 = pd.Series(img_call, name = "post_num")
	df = pd.concat([s1,s2, s3,s4,s5,s6,s7,s8,s9,s10, s11, s13], axis=1)

	#append df to existing csv (create one before running)
	df.to_csv('content.csv', mode='a', header=False)
	print("Data appended successfully.")

def all_first_images(hashtag):

	time.sleep(1)

	html = driver.page_source
	# img_url = driver.current_url
	# img_call = img_url.split('/')[-2]
	soup = bs(html, 'html.parser')
    
	imgs = {src for src in soup.findAll('a', class_="x1i10hfl xjbqb8w x1ejq31n xd10rxx x1sy0etr x17r0tee x972fbf xcfux6l x1qhh985 xm0m39n x9f619 x1ypdohk xt0psk2 xe8uvvx xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x16tdsg8 x1hl2dhg xggy1nq x1a2a7pz x4gyw5p _a6hd")}
	i=0
	time.sleep(5)
	for img in imgs:
		i += 1
		# print('img in for loop', img)
		# href = soup.href
		img_pCall = img['href'] #figure out how to get attribute
		img_pCall = img_pCall.split('/')[-2]
		print('href', img_pCall)
		img_src = img.find("img")['src']
		img_alt = img.find("img")['alt']
		jpg = urllib.request.urlretrieve(img_src, hashtag + '/thumbnail_' + img_pCall + '_{}.jpg'.format(str(i)))
		s0 = pd.Series(jpg[0], name='first_image')
		s1 = pd.Series(hashtag + '/'+ img_pCall + ".jpg", name='file_path')
		s2 = pd.Series(img_alt, name="img_alt")
		df = pd.concat([s0, s1,s2], axis=1)
	    #append df to existing csv (create one before running)
		df.to_csv('content_swoop.csv', mode='a', header=False)
	print("First images' src/alt appended successfully.")	


	#############################################

def create_folder(fldr_name):
	# check if folder corresponding to hashtag exists in your dir
	if(os.path.isdir(fldr_name) == False):
		# Create folder
		os.mkdir(str(fldr_name))
	print(fldr_name, "created successfully")

		


def download_allposts(hashtag, c):

	#make sure you're in the correct folder anytime
	# parent_path = os.getcwd()
	
    #folder names to test different functions
	hashtest = str(hashtag)+'01'
	only_imgs = str(hashtag)+'_sweep'
	# check if folder corresponding to hashtag exists in your dir
	create_folder(hashtag)
	create_folder(hashtest)
	create_folder(only_imgs)

    # get all the thumbnails and alt info form the landing page
	all_first_images(hashtag)
	
	#open the first post to begin clicking throuh imaages
	first_post()
	
	
	# Check if the first Post contains multiple images or videos; # multiple_images = nested_check()
    # if nested chexk exists, then the are multiple images in the post
	if nested_check():
		print("Post has multiple images/videos")
		post_count = 1
		nescheck = nested_check()
		count_img = 0
		# imagesrc = []
		

		while nescheck:
			WebDriverWait(driver, 1.5)
			elem_img = driver.find_element(By.CLASS_NAME,'_aagv').get_attribute('src') #_aagw

			save_carousel_post(hashtag, hashtag+'/'+'post_' + str(post_count)+'_image_0'+str(count_img))
			print("added multiple image post")
			nescheck.click()
			nescheck = nested_check()
			count_img +=1
   

	else:
		# this goes to the function above, which uses BS to look through the html and find the source image and other things
        # making my own function here for testing
		# save_content('_aagv', hashtag+'/'+'content1') #_aagw
		print("We have one image/video")
		post_count = 1
		# elem_img = driver.find_element(By.CLASS_NAME,'_aagv').get_attribute('src')
		save_sngl_content_post(hashtag, hashtag+'/'+'post_' + str(post_count)+'_image0')
		#hashtag+'/'+'_post_' +	str(c)+'_image'+str(count_img)
        # save_PostContent(hashtag, hashtag +'_postcnt_'+str(post_count))
		print('added single image post')
		post_count +=1
		# next_post().click()


	# c = 2
	
	# while(True):
	while(c <= 1000):
		print("c: ", c)
		print("current post count", post_count)
		next_el = next_post()
		print("next_el", next_el)
		
		if next_el != False:
		# if next_el != 0:
			next_el.click()
			print("onto the next_el")
			count_img = 0

			time.sleep(3)
			
			# try:
			multiple_images = nested_check()
			
			if multiple_images:
				print('next post has multiple images/vids')
				nescheck = multiple_images

				img_url = driver.current_url
				img_call = img_url.split('/')[-2]
				print("New URL:", img_call)

    			#get the first image of each carousel post because the above code skips it somehow
				first_image(hashtag, img_call, c)
                #aggregate the number of images in the post
				count_img +=1
					
				#as long as there are nested images in the carasel ...
				while nescheck:
					save_carousel_post(hashtag, hashtag+'/'+'post_' +
								str(c)+'_image_0'+str(count_img))						
	                #aggregate the number of images in the post
					count_img += 1
					
					# identify if there are more images in the post 
					nescheck = nested_check()

			else:
				#save the post
				save_sngl_content_post(hashtag, hashtag+ '/'+'post_' +
								str(c))
					# save_PostContent(hashtag, hashtag+'_image'+str(c))
				print("saved single post")
				post_count +=1
				# next_post().click()
				# next_post()


		
		else:
			break
		
		c += 1
	# driver.close()
	return


def scroll(hashtag, c):
	time.sleep(10)
	scrolldown = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var scrolldown = document.body.scrollHeight;return scrolldown;")
	match = False

	while(match == False):
		last_count = scrolldown
		time.sleep(5)
		# download_allposts(hashtag)
		scrolldown = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var scrolldown=document.body.scrollHeight;return scrolldown;")
		download_allposts(hashtag, c)
		time.sleep(1)
		if last_count == scrolldown:
			# download_allposts(hashtag)
			match = True
	driver.close()


if __name__ == "__main__":
	path()

	#account tests
	# account = "apartados8km"
	# account = 'feminicidiocdmx_' #start at c = 54
	
	# account = "siwapazyjusticia"
	# account = "abogadafemina"
	# account = "redpsicofem.jrz"
	# account = "colectivoyositecreo"
	# account = "EstadoFeminicida"

	# url = "https://www.instagram.com/" + account
	# login(url)
	# start c at 2 because the scroll function automatically pulls information for the first post
	# scroll(account, 2)


	#different variations of the URL to get around different stages of getting kicked out of instagram
	# https://www.instagram.com/siwapazyjusticia
	# https://www.instagram.com/abogadafemina
	# https://www.instagram.com/colectivoyositecreo
	



    ####

 
