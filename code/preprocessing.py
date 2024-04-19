import numpy as np
# import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image

import matplotlib.pyplot as plt


def preprocess():

    '''
    This function crops the images to only include the license plate
    '''

    for i in range(1,5001):
        num_zeroes = 6 - len(str(i)) # file name purposes
        zeroes = ""
        for j in range(num_zeroes):
            zeroes += "0" # file name purposes
        img_fp = "data/cars-br/img_" + zeroes + str(i) + ".jpg"
        txt_fp = "data/cars-br/img_" + zeroes + str(i) + ".txt"
        im = Image.open(img_fp)

        with open(txt_fp, "r") as txt_file:
            file_contents = txt_file.readlines()
            coords = file_contents[3].split(" ") # Corner coordinates of the license plate

            # add padding of 5 pixels just in case :)
            # get coordinates to crop image
            left = int(coords[1].split(",")[0]) - 5
            upper = int(coords[1].split(",")[1]) - 5
            right = int(coords[3].split(",")[0]) + 5
            bottom = int(coords[3].split(",")[1]) + 5
        im = im.crop((left, upper, right, bottom))
        new_fp = "data/cars-br-cropped/img_" + zeroes + str(i) + ".jpg"
        im.save(new_fp)

def get_labels():
    '''
    This function reads the txt files and extracts the license plate values.
    Our labels are just the raw text values of the license plates because we are using CTC loss.
    '''
    plates = []
    for i in range(1,5001):
        num_zeroes = 6 - len(str(i))
        zeroes = ""
        for j in range(num_zeroes):
            zeroes += "0" # file name purposes
        txt_fp = "data/cars-br/img_" + zeroes + str(i) + ".txt"
        # open txt file with car information
        with open(txt_fp, "r") as txt_file:
            file_contents = txt_file.readlines()
            plate_id = file_contents[1].split(" ")
            # get the plate values from the txt files
            plate = plate_id[1]
            # add plate values to labels array
            plates.append(plate)
    return np.array(plates)

def get_inputs():
    # TODO: Implement Spatial Transformer Layer proprocess to improve model performance
    ## 24 is the width, 94 is the height of the image
    # We chose these dimensions to be consistent with the basis paper
    images = np.zeros(shape=(5000, 24, 94, 3)) # 5000 images, 24x94 pixels, 3 channels
    for i in range(1,5001):
        num_zeroes = 6 - len(str(i))
        zeroes = ""
        for j in range(num_zeroes):
            zeroes += "0" # file name purposes
        img_fp = "data/cars-br-cropped/img_" + zeroes + str(i) + ".jpg"
        # open image and convert to RGB values
        im = Image.open(img_fp)
        im = im.convert('RGB')
        # Resize to be consistent with paper
        im = im.resize((94,24))
        im_array = np.array(im, dtype='float64')
        # normalize pixel values
        im_array /= 255.0
        # add to input array
        images[i-1] = im_array
    return images


# TODO: call preprocess() if the cropped images are not already saved
# preprocess()

def get_data():
    inputs = get_inputs()
    labels = get_labels()
    return inputs, labels

