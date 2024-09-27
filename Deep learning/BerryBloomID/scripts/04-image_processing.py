#!/usr/bin/python3

#Image processing
#---
#Script to process images for input into the ViT model.


#load libraries
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from datasets import Dataset, Features, ClassLabel, Value, Image as DatasetImage
from transformers import ViTImageProcessor

#define global variables
DIR = "/mnt/c/Users/dstra/Nextcloud/Documents/Computers/Code/Github repos/Deep learning/BertaBloomID" #working directory

#load data
train_data = pd.read_csv(f"{DIR}/data/train_data.csv")
test_data = pd.read_csv(f"{DIR}/data/test_data.csv")


#Preprocess image data
#---
#allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

#define the dataset object to contain the processed images
features = Features({
    'image_path': Value(dtype = 'string'),
    'label': ClassLabel(names = train_data['label'].unique().tolist())
})

#define function to process the images
def process_images(example):
    #define the current samples image path
    image_path = example['image_path']

    #open the current samples image
    image = Image.open(image_path).convert("RGB")

    #process the image for input into the ViT model
    pixel_values = processor(image, return_tensors = "pt")['pixel_values'].squeeze()
    example['pixel_values'] = pixel_values
    del image_path
    return example

#initialize the ViT model processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

#create the training dataset
train_dataset = Dataset.from_pandas(train_data, features = features)
train_dataset = train_dataset.map(process_images, batched = False)

#create the testing dataset
test_dataset = Dataset.from_pandas(test_data, features = features)
test_dataset = test_dataset.map(process_images, batched = False)

#save the datasets to disk
train_dataset.save_to_disk(f"{DIR}/data/train_dataset")
test_dataset.save_to_disk(f"{DIR}/data/test_dataset")
