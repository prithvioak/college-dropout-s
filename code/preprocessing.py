import numpy as np
# import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image
import cv2

import matplotlib.pyplot as plt

ALL_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_MAP = {char: idx for idx, char in enumerate(ALL_CHARS)}
REVERSE_CHAR_MAP = {idx: char for idx, char in enumerate(ALL_CHARS)}

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
            left = int(coords[1].split(",")[0])
            upper = int(coords[1].split(",")[1])
            right = int(coords[3].split(",")[0])
            bottom = int(coords[3].split(",")[1])
        im = im.crop((left, upper, right, bottom))
        new_fp = "data/cars-br-cropped/img_" + zeroes + str(i) + ".jpg"
        im.save(new_fp)

def get_labels():
    '''
    This function reads the txt files and extracts the license plate values.
    Our labels are just the raw text values of the license plates because we are using CTC loss.
    '''
    #### CHANGE THE ABOVE DESCRIPTION
    plates = []
    ## Out one-hot encoded data is of shape (5000, 7, 37),
    ## where 5000 is the number of images, 7 is the number of characters in the license plate,
    ## and 37 is the number of possible characters.
    onehot = np.zeros((5000, 7, len(ALL_CHARS)))
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
            # # add plate values to labels array
            plate = plate[:-1]
            # # convert all characters to their unique values gathered from the dictionary
            # new_plate = []
            # for i in plate:
            #     new_plate.append(CHAR_MAP[i])
            plates.append(plate)
        ## one-hot encode the labels
        if len(plate) != 7:
            print(list(plate))
            raise ValueError("Plate is not 7 characters long")
        
        for j, char in enumerate(plate):
            onehot[i-1, j, CHAR_MAP[char]] = 1
    # print("first 5 labels: ", [(plates[i], onehot[i]) for i in range(5)])
    return np.array(onehot)

def get_labels_lprnet():
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
            plate = list(plate)[:-1]
            # convert all characters to their unique values gathered from the dictionary
            new_plate = []
            for i in plate:
                new_plate.append(CHAR_MAP[i])
            plates.append(new_plate)
    return np.array(plates)

def get_inputs_lprnet():
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

def get_segmented_images():
    '''
    Returns a list (length 5000) of lists (length 7) of characters where each character is a 3D numpy array of dimensions 32x24x3
    '''
    total_characters = np.zeros((5000, 7, 32, 24, 3))

    for i in range(1,5001):
        num_zeroes = 6 - len(str(i))
        zeroes = ""
        for j in range(num_zeroes):
            zeroes += "0"
        img_fp = "data/cars-br-cropped/img_" + zeroes + str(i) + ".jpg"
        image = cv2.imread(img_fp)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find top 7 contours by area and sort them from left to right
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:7]
        contours = sorted(biggest_contours, key=lambda c: cv2.boundingRect(c)[0])

        # Loop through contours
        for j, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
            # Extract character region
            # offset = 1
            # if y-offset < 0:
            #     y = 0
            # else:
            #     y -= offset
            # if x-offset < 0:
            #     x = 0
            # else:
            #     x -= 5
            # if x+w+offset > image.shape[1]:
            #     w = image.shape[1] - x
            # else:
            #     w += offset
            # if y+h+offset > image.shape[0]:
            #     h = image.shape[0] - y
            # else:
            #     h += offset

            character_region = thresh[y:y+h, x:x+w]

            # character_region = cv2.resize(character_region, (24, 32))
            # resize character region to 32x24 by padding, not changing aspect ratio
            character_region = cv2.resize(character_region, (24, 32), interpolation=cv2.INTER_AREA)
    
            # Convert character region to RGB tensor
            character_tensor = cv2.cvtColor(character_region, cv2.COLOR_BGR2RGB)
            
            # character_tensor = tf.convert_to_tensor(character_tensor)
            # Append to the list of characters
            total_characters[i-1, j] = character_tensor
        if i==1:
            # Uncomment to visualize the segmented characters
            # for j, character in enumerate(total_characters[i-1]):
            #     cv2.imshow('Character {}'.format(j+1), character)
            # cv2.imshow('Segmented Characters', image)
            
            cv2.imshow('Segmented Characters', np.hstack([np.pad(character, ((10, 10), (10, 10), (0, 0)), constant_values=0.90) for character in total_characters[i-1]]))


            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    # convert to Tensor
    total_characters = tf.convert_to_tensor(total_characters)
    return total_characters 



# TODO: call preprocess() if the cropped images are not already saved
# preprocess()

def get_data_segmented():
    inputs = get_segmented_images()
    labels = get_labels()
    return inputs, labels

def get_data_lprnet():
    inputs = get_inputs_lprnet()
    labels = get_labels_lprnet()
    return inputs, labels

# FOR TESTING PURPOSES
# get_segmented_images()
