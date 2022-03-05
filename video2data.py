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
from skimage.measure import label
import scipy.ndimage as nd
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
from TikTokApi import TikTokApi

second_to_info = {"5": "Name", "6": "Age", "7": ["Height", "Gender"], "13": "Location", "16":"Profession"}

# Note: This works well!
def convert_video_to_image(VIDEO_PATH, IMAGE_PATH):
  KPS = 1
  EXTENSION = ".png"
  cap = cv2.VideoCapture(BASIC_PATH + "videos/" + VIDEO_PATH)
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

def get_ncc(img):
  _, ncc = label(img, connectivity=2, return_num=True)
  return ncc

def get_ccc(img, i):
  mask_log_i = get_mask_log(img, i)
  ncc_i = get_ncc(mask_log_i)

  mask_log_i_2 = get_mask_log(img, i - 2)
  ncc_i_2 = get_ncc(mask_log_i_2)

  return 1 - ncc_i / ncc_i_2


def get_stc(img, i):
  mask_log_1 = get_mask_log(img, 1)
  ncc_1 = get_ncc(mask_log_1)

  mask_log_i = get_mask_log(img, i)
  ncc_i = get_ncc(mask_log_i)

  mask_log_i_2 = get_mask_log(img, i - 2)
  ncc_i_2 = get_ncc(mask_log_i_2)

  return ncc_i / ncc_1 * abs(ncc_i_2 - ncc_i)


def get_mask_log(img, i):
  sigma = 0.3 * ((i - 1) * 0.5 - 1) + 0.8
  log = nd.gaussian_laplace(img, sigma)
  log[log < 0] = 0
  log = np.float32(log)
  log = cv2.cvtColor(log, cv2.COLOR_BGR2GRAY)
  log = log.astype('uint8')
  _, log = cv2.threshold(log, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

  return log


def screentone_removal(img):
  i = 3
  ccc_max = 0
  i_log = 1

  ALPHA = 1
  BETA = 0.8

  stc_i = get_stc(img, i)

  while stc_i > ALPHA:
    ccc_i = get_ccc(img, i)
    if ccc_i >= BETA * ccc_max:
      i_log = i
      ccc_max = max(ccc_i, ccc_max)
    i += 2

    stc_i = get_stc(img, i)

  i_log += 4
  i_base = min(int(i / 2), i_log)
  mask_rm = cv2.bitwise_or(get_mask_log(img, i_log), get_mask_log(img, i_base))

  return i_log, i_base, mask_rm


# Note: This does not do a great job yet of extracting text in images - still debugging this
def convert_image_to_text(IMAGE_PATH, TEXT_PATH):
  user_dict = {}
  # iterating the images inside the folder
  for imageName in os.listdir(IMAGE_PATH):
    if imageName.endswith(".png"):
      imageName_raw = imageName.replace(".png", "")
      inputPath = os.path.join(IMAGE_PATH, imageName)
      img = cv2.imread(inputPath)
      gray = cv2.GaussianBlur(img, (3, 3), 0)
      cv2.fastNlMeansDenoising(gray, gray, 20)
      img1 = gray * 1.0
      i_log, i_base, mask_rm = screentone_removal(img1)

      # Perform OCR
      if imageName_raw in second_to_info:
        info_type = second_to_info[imageName_raw]
        if isinstance(info_type, list):
          for item in info_type:
            user_dict[item] = pt.image_to_string(mask_rm, lang='eng', config='--psm 6')
        else:
          user_dict[info_type] = pt.image_to_string(mask_rm, lang='eng', config='--psm 6')

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
VIDEO_PATHS  = BASIC_PATH + "videos/"
IMAGE_PATH = BASIC_PATH + "images/"
TEXT_PATH =  BASIC_PATH + "outputFile.txt"
list_of_trends = ["6829082401130416897"]

# Grab video links
actual = pd.read_csv(BASIC_PATH + "actual_data.csv")
actual = actual.rename(columns={'Name ': 'Name'})
api = TikTokApi(custom_verify_fp=os.environ.get("verifyFp", None))
"""
for row in actual.iterrows():
  video = api.video(id=int(row[1].Id))
  video_data = video.bytes()
  with open(f'videos/{row[0]}.mp4', "wb") as out_file:
    out_file.write(video_data)
"""
#get_unique_users_by_trend(BASIC_PATH, list_of_trends)
test = []
for i, videoName in enumerate(os.listdir(VIDEO_PATHS)):
  convert_video_to_image(videoName, IMAGE_PATH)
  user_dict = convert_image_to_text(IMAGE_PATH, TEXT_PATH)
  df_test = pd.DataFrame.from_dict(user_dict, orient='index').T
  print(df_test)
  test.append(df_test)

test = pd.concat(test)
actual = actual.drop(columns=['Link', 'Id']).reindex(test.columns, axis=1)
print(actual.head(6).eq(test.values).mean())


#dict = clean_dict(user_dict)






