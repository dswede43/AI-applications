#!/usr/bin/python3

#Data splitting
#---
#Script to split the image data into train and test sets.


#Table of contents:
#---
#1. Organize the image path data
#2. Create the training and test data


#load libraries
import os
import numpy as np
import pandas as pd

#define global variables
DIR = "/mnt/c/Users/dstra/Nextcloud/Documents/Computers/Code/Github repos/Deep learning/BertaBloomID" #working directory
IMAGE_DIR = "/mnt/g/Machine_learning/BertaBloomID_images" #direcory to download images
SPLIT_RATIO = 0.8 #train test split ratio


#1. Organize the image path data
#---
print("Organizing the image path data.")

#define the list of plant species
plants = os.listdir(IMAGE_DIR)

label = 0
image_data = pd.DataFrame(columns = ['image_path','common_name','label'])
for plant in plants:
    #obtain all downloaded image paths
    image_dir = f"{IMAGE_DIR}/{plant}"
    images = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, image) for image in images]
    
    #organize and append data into growing data frame
    image_dict = {'image_path': image_paths, 'common_name': plant, 'label': label}
    image_df = pd.DataFrame(image_dict)
    image_data = pd.concat([image_data, image_df])
    
    #update the class label
    label += 1

#save the data as CSV
image_data.to_csv(f"{DIR}/data/image_data.csv", index = False)


#2. Create the training and test data
#---
print("Splitting the image data into train and test sets.")

#set the seed
np.random.seed(42)

#randomly shuffle the data frame indices
index_length = image_data.shape[0]
indices = np.arange(index_length)
np.random.shuffle(indices)

#define the split index
split_index = int(index_length * SPLIT_RATIO)

#define the train and test indices
train_indices = indices[:split_index]
test_indices = indices[split_index:]

#split the data into train and test sets
train_data = image_data.iloc[train_indices]
train_data = train_data[['image_path','label']]

test_data = image_data.iloc[test_indices]
test_data = test_data[['image_path','label']]

#save the data as CSV
train_data.to_csv(f"{DIR}/data/train_data.csv", index = False)
test_data.to_csv(f"{DIR}/data/test_data.csv", index = False)
