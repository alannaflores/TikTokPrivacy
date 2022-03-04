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
    if not ret or curr_frame > 400:
      break
    if curr_frame % hop == 0:
      name = IMAGE_PATH + "_" + str(curr_frame) + EXTENSION
      cv2.imwrite(name, frame)
    curr_frame += 1
  cap.release()


# Note: This does not do a great job yet of extracting text in images - still debugging this
def convert_image_to_text(IMAGE_PATH, TEXT_PATH):
  text = " "
  # iterating the images inside the folder
  for imageName in os.listdir(IMAGE_PATH):
    inputPath = os.path.join(IMAGE_PATH, imageName)
    image = cv2.imread(inputPath)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([100, 175, 110])
    mask = cv2.inRange(hsv, lower, upper)

    # Morph close to connect individual text into a single contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find rotated bounding box then perspective transform
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (36, 255, 12), 2)
    warped = four_point_transform(255 - mask, box.reshape(4, 2))

    # Perform OCR
    text += pt.image_to_string(warped, lang='eng', config='--psm 6')

    file1 = open(TEXT_PATH, "a+")

    # providing the name of the image
    file1.write(imageName + "\n")

    # providing the content in the image
    file1.write(text + "\n")
    file1.close()

  file2 = open(TEXT_PATH, 'r')
  file2.close()
  return text

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
def convert_text_to_dataframe(text, TEXT_PATH):
  # Make sample text for testing
  sample_text = ["Here's a song to get to know about me", "Sofia Bella", "24 5'2 She/queen", "Las Vegas", "Teacher"]

  for item in sample_text:
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
    tag = api.hashtag(name=trend)
    users = []
    # Can't really go above looking at the users for more than 1000
    # because of limited computational capacity (tiktok API times you out; we would need to spin up proxies)
    for video in tag.videos(count=5000):
      users.append(video.author.user_id)
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
convert_video_to_image(VIDEO_PATH, IMAGE_PATH)
text = convert_image_to_text(IMAGE_PATH, TEXT_PATH)
dict = convert_text_to_dataframe(text, TEXT_PATH)

list_of_trends = ["adhd", "aboutmechallenge"]
api = TikTokApi(custom_verify_fp=os.environ.get("verifyFp", None))
get_unique_users_by_trend(BASIC_PATH, list_of_trends)




