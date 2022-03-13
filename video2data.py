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

import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

second_to_info = {"5": "Name", "6": "Age", "7": ["Height", "Gender"], "13": "Location", "16": "Profession"}


def convert_video_to_image(VIDEO_PATH, IMAGE_PATH):
    KPS = 1
    EXTENSION = ".png"
    cap = cv2.VideoCapture(BASIC_PATH + "videos/" + VIDEO_PATH)
    # how frequently a video frame should be converted to image
    hop = round(cap.get(cv2.CAP_PROP_FPS) / KPS)
    curr_frame = 0

    while (True):
        ret, frame = cap.read()
        if not ret or curr_frame > 490:
            break
        if curr_frame % hop == 0:
            name = IMAGE_PATH + VIDEO_PATH.replace(".mp4", "") + str(int(curr_frame / 30)) + EXTENSION
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
            img = keras_ocr.tools.read(inputPath)
            gray = cv2.GaussianBlur(img, (3, 3), 0)
            cv2.fastNlMeansDenoising(gray, gray, 20)
            img1 = gray * 1.0
            i_log, i_base, mask_rm = screentone_removal(img1)

            # Perform OCR
            if imageName_raw in second_to_info:
                info_type = second_to_info[imageName_raw]
                if isinstance(info_type, list):
                    for item in info_type:
                        user_dict[item] = pt.image_to_string(img, lang='eng', config='--psm 6')
                else:
                    user_dict[info_type] = pt.image_to_string(img, lang='eng', config='--psm 6')
                    print(user_dict[info_type])

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
    # for item in sample_text:
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
VIDEO_PATHS = BASIC_PATH + "videos/"
IMAGE_PATH = BASIC_PATH + "images/"
TEXT_PATH = BASIC_PATH + "outputFile.txt"
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
# get_unique_users_by_trend(BASIC_PATH, list_of_trends)
test = []

actual = actual.drop(columns=['Link', 'Id']) #.reindex(test.columns, axis=1)
second_to_info = {"5": "Name", "6": "Age", "7": ["Height", "Gender"], "13": "Location", "16": "Profession"}


def load_dataset():
    # initialize the list of data and labels
    data = []
    labels = []
    l = os.listdir(VIDEO_PATHS)
    videos = sorted(l, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    for i, videoName in enumerate(videos):
       #convert_video_to_image(videoName, IMAGE_PATH)
        # parse the label and image from the row
        row = actual.iloc[i]
        for key in second_to_info:
            image = "images/" + videoName.replace(".mp4", "") + key + ".png"
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


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 50
INIT_LR = 1e-1
BS = 128
# load the A-Z and MNIST datasets, respectively
print("[INFO] loading datasets...")
images, labels = load_dataset()
characters = set(char for label in labels for char in str(label))

correct = 0
total = 0
for i in range(len(images)):
    img = cv2.imread(images[i])
    gray = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.fastNlMeansDenoising(gray, gray, 20)
    img1 = gray * 1.0
    i_log, i_base, mask_rm = screentone_removal(img1)
    predicted = pt.image_to_string(mask_rm, lang='eng', config='--psm 6')
    actual = labels[i]
    total += 1
    if predicted == actual:
        correct += 1
    print(correct)
    print(total)
print(correct)
print(total)


# Batch size for training and validation
batch_size = 1
img_height = 400
img_width = 100

# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4

max_length = max([len(str(label)) for label in labels])

# Mapping characters to integers
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


# Splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))


def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((list(x_valid), list(y_valid)))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

"""
_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
    images = batch["image"]
    labels = batch["label"]
    for i in range(batch_size):
        img = (images[i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.show()
"""

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        print(y_true)
        print(y_pred)
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


# Get the model
model = build_model()
model.summary()

print("im here")
epochs = 2
early_stopping_patience = 10
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

print("now im here")

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)

print("im here")
prediction_model.summary()

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


#  Let's check results on some validation samples
for batch in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.show()



# user_dict = convert_image_to_text(IMAGE_PATH, TEXT_PATH)
# df_test = pd.DataFrame.from_dict(user_dict, orient='index').T
# print(df_test)
# test.append(df_test)

# test = pd.concat(test)
# test.to_csv("This_one.csv")


# print(actual.head(6).eq(test.values).mean())


# dict = clean_dict(user_dict)
