#!/usr/bin/python3

#Image cleaning
#---
#Script to find and remove corrupted images.


#load libraries
import os
import shutil
from PIL import Image, UnidentifiedImageError

#define global variables
IMAGE_DIR = "/mnt/g/Machine_learning/BertaBloomID_images" #direcory to download images
CORRUPT_DIR = "/mnt/g/Machine_learning/corrupt_images" #directory for corrupted images


#define the list of plant species
plants = os.listdir(IMAGE_DIR)


for plant in plants:
    #obtain all downloaded image paths
    image_dir = f"{IMAGE_DIR}/{plant}"
    images = os.listdir(image_dir)

    for image in images:
        image_path = os.path.join(image_dir, image)
        print(f"Verifying image: {image_path}")
        try:
            with Image.open(image_path) as img:
                img.verify()
        except (UnidentifiedImageError, IOError) as e:
            print(f"Corrupted image detected: {image_path}")
            shutil.move(image_path, os.path.join(CORRUPT_DIR, plant))
