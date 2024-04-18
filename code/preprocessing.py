import numpy as np
# import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image

import matplotlib.pyplot as plt

def preprocess():
    for i in range(1,5001):
        num_zeroes = 6 - len(str(i))
        zeroes = ""
        for j in range(num_zeroes):
            zeroes += "0" # file name purposes
        img_fp = "data/cars-br/img_" + zeroes + str(i) + ".jpg"
        txt_fp = "data/cars-br/img_" + zeroes + str(i) + ".txt"
        im = Image.open(img_fp)

        with open(txt_fp, "r") as txt_file:
            file_contents = txt_file.readlines()
            coords = file_contents[3].split(" ")
            # add padding of 10 pixels just in case :)
            # get coordinates to crop image
            left = int(coords[1].split(",")[0]) - 5
            upper = int(coords[1].split(",")[1]) - 5
            right = int(coords[3].split(",")[0]) + 5
            bottom = int(coords[3].split(",")[1]) + 5
        im = im.crop((left, upper, right, bottom))
        new_fp = "data/cars-br-cropped/img_" + zeroes + str(i) + ".jpg"
        im.save(new_fp)

preprocess()