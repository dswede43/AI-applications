#!/usr/bin/python3

#Image scraping
#---
#Script to scrape images from Bing.


#load libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
from icrawler.builtin import BingImageCrawler
from concurrent.futures import ThreadPoolExecutor

#define global variables
DIR = "/mnt/c/Users/dstra/Nextcloud/Documents/Computers/Code/Github repos/Deep learning/BertaBloomID" #working directory
IMAGE_DIR = "/mnt/g/Machine_learning/BertaBloomID_images" #direcory to download images
NUM_IMAGES = 50 #number of images to download of each species
NUM_THREADS = 8 #number of threads for multi-processing

#load data
plants_df = pd.read_csv(f"{DIR}/data/mono-dicots_alberta.csv")

define the unique list of plant species
plants = plants_df['scientific_name'].unique()
plants = [' '.join(plant.split()[:2]) for plant in plants]
plants = list(set(plants))


#Download and process images for ViT input
#---
#define function to resize images
def resize_image(image_path, output_size = (224, 224)):
    try:
        img = Image.open(image_path)
        img = img.resize(output_size)
        img.save(image_path)
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")

#define function to convert images to .jpg format
def convert_to_jpg(image_path):
    try:
        #check if image is a .jpg
        if not image_path.endswith('.jpg'):
            #open the image file
            img = Image.open(image_path)

            #convert image to RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')

            #create and save new .jpg filename
            new_image_path = os.path.splitext(image_path)[0] + '.jpg'
            img.save(new_image_path, 'JPEG')
            
            #remove the original non-JPG file
            os.remove(image_path)

            #return the new image path
            return new_image_path

        #return the original file path if already a .jpg
        return image_path  
    except Exception as e:
        print(f"Error converting image {image_path} to .jpg: {e}")

#define function to download Bing images
def download_and_process_images(plant, output_dir, max_num = 10, num_threads = 4):
    #create a directory for the current plant if it does not exist
    if not os.path.exists(f"{output_dir}/{plant}"):
        os.makedirs(f"{output_dir}/{plant}")

    #initialize the Bing image scraper
    bing_crawler = BingImageCrawler(
        downloader_threads = num_threads,
        storage = {'root_dir': f"{output_dir}/{plant}"})

    #scrape the images
    bing_crawler.crawl(
        keyword = f"{plant} plant",
        filters = None,
        offset = 0,
        max_num = max_num)

    #obtain all downloaded image paths
    image_dir = f"{output_dir}/{plant}"
    images = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, image) for image in images]

    #resize and convret to .jpg for each image
    for image_path in image_paths:
        resize_image(image_path)
        convert_to_jpg(image_path)

#download and process each image for all plant species
for plant in plants:
    download_and_process_images(plant,
        output_dir = IMAGE_DIR,
        max_num = NUM_IMAGES,
        num_threads = NUM_THREADS)
