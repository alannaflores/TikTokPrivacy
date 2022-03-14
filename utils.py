from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
import cv2
import os
from video2data import convert_video_to_image

BASIC_PATH = "/Users/alanna/Github/TikTokPrivacy/"
VIDEO_PATH = 'sample_video.mp4'
VIDEO_PATHS = BASIC_PATH + "videos/"
actual = pd.read_csv(BASIC_PATH + "actual_data.csv")
actual = actual.rename(columns={'Name ': 'Name'})
actual = actual.drop(columns=['Link', 'Id'])
second_to_info = {"5": "Name", "6": "Age", "7": ["Height", "Gender"], "13": "Location", "16": "Profession"}

def load_dataset():
    # initialize the list of data and labels
    data = []
    labels = []
    l = os.listdir(VIDEO_PATHS)
    videos = sorted(l, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    for i, videoName in enumerate(videos):
        # parse the label and image from the row
        row = actual.iloc[i]
        for key in second_to_info:
            image = "images/" + videoName.replace(".mp4", "") + key + ".png"
            image = cv2.imread(image)
            if isinstance(second_to_info[key], list):
                for item in second_to_info[key]:
                    label = row[item]
                    data.append(image)
                    labels.append(label)
            else:

                label = row[second_to_info[key]]
                data.append(image)
                labels.append(label)

    # return a 2-tuple of the A-Z data and labels
    return (data, labels)

