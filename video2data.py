"""
A Four Part Pipeline to Extracting Tiktok User Data
"""
import os
import cv2
import time
import geonamescache
import pandas as pd
import numpy as np
import pytesseract as pt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
from TikTokApi import TikTokApi

second_to_info = {"5": "Name", "6": "Age", "7": ["Height", "Gender"], "13": "Location", "16":"Profession"}

# Note: This works well!
def convert_video_to_image(VIDEO_PATH, IMAGE_PATH):
  KPS = 1
  EXTENSION = ".png"
  cap = cv2.VideoCapture(BASIC_PATH + VIDEO_PATH)
  # how frequently a video frame should be converted to image
  hop = round(cap.get(cv2.CAP_PROP_FPS) / KPS)
  curr_frame = 0

  while(True):
    ret, frame = cap.read()
    if not ret or curr_frame > 490:
      break
    if curr_frame % hop == 0:
      name = IMAGE_PATH + str(int(curr_frame/30)) + EXTENSION
      cv2.imwrite(name, frame)
    curr_frame += 1
  cap.release()


# Note: This does not do a great job yet of extracting text in images - still debugging this
def convert_image_to_text(IMAGE_PATH, TEXT_PATH):
  user_dict = {}
  # iterating the images inside the folder
  for imageName in os.listdir(IMAGE_PATH):
    imageName_raw = imageName.replace(".png", "")
    inputPath = os.path.join(IMAGE_PATH, imageName)
    image = cv2.imread(inputPath)

    # Perform OCR
    if imageName_raw in second_to_info:
      info_type = second_to_info[imageName_raw]
      if isinstance(info_type, list):
        for item in info_type:
          user_dict[item] = pt.image_to_string(image, lang='eng', config='--psm 6')
      else:
        user_dict[info_type] = pt.image_to_string(image, lang='eng', config='--psm 6')

  return user_dict

"""
Once the text is detected we will need to identify which words correspond with 
I think we need to make a list of all the possibilities for these things:
-Locations
-Age 
-Profession
-Height 
-Gender - she first then they then he 

and then check if a word we have extracted from the images exists in these lists
of possibilities 

I've started below by creating a list of all possible locations
"""
def clean_dict(user_dict):
  #for item in sample_text:
  # Age
  # Profession
  # Height
  # Gender

  # Locations
  gc = geonamescache.GeonamesCache()
  countries = list(pd.DataFrame(gc.get_countries()).T["name"])
  states = list(pd.DataFrame(gc.get_us_states()).T["name"])
  cities = list(pd.DataFrame(gc.get_cities()).T["name"])
  locations_list = countries + states + cities


def get_unique_users_by_trend(BASIC_PATH, list_of_trends):
  dfs = []
  for trend in list_of_trends:
    tag = api.sound(id=trend)
    users = []
    # Can't really go above looking at the users for more than 1000
    # because of limited computational capacity (tiktok API times you out; we would need to spin up proxies)
    for i, video in enumerate(tag.videos(count=5)):
      users.append(video.author.user_id)
      video_data = video.bytes()
      with open(f'video{i}.mp4', "wb") as out_file:
        out_file.write(video_data)
    dfs.append(users)
    unique_ids = len(users)
    print(trend + ": " + str(unique_ids))
  intersection_ids = list(set.intersection(*map(set, dfs)))
  total_unique_ids = len(intersection_ids)
  print("Unique users who have done all trends: " + str(total_unique_ids))

BASIC_PATH = "/Users/alanna/Github/TikTokPrivacy/"
VIDEO_PATH = 'sample_video.mp4'
IMAGE_PATH = BASIC_PATH + "images/"
TEXT_PATH =  BASIC_PATH + "outputFile.txt"
list_of_trends = ["6829082401130416897"]
#api = TikTokApi(custom_verify_fp=os.environ.get("verifyFp", None))
#get_unique_users_by_trend(BASIC_PATH, list_of_trends)
convert_video_to_image(VIDEO_PATH, IMAGE_PATH)
user_dict = convert_image_to_text(IMAGE_PATH, TEXT_PATH)
print(user_dict)
#dict = clean_dict(user_dict)






