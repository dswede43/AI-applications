#!/usr/bin/python3

#Image dataset using ImageFolder
#---
#Script to process images as a dataset using ImageFolder for input into the ViT model.


#load libraries
import os
import shutil
import pandas as pd
from datasets import load_dataset

#define global variables
DIR = "/mnt/c/Users/dstra/Nextcloud/Documents/Computers/Code/Github repos/Deep learning/BertaBloomID" #working directory
IMAGE_DIR = "/mnt/g/Machine_learning/BertaBloomID_images" #direcory to download images

#load data
train_data = pd.read_csv(f"{DIR}/data/train_data.csv")
test_data = pd.read_csv(f"{DIR}/data/test_data.csv")




def image_folders(image_paths, split = "train"):
    for image_path in image_paths:
        plant = image_path.split("/")[5]
        
        if not os.path.exists(f"{IMAGE_DIR}/{split}/{plant}"):
            os.makedirs(f"{IMAGE_DIR}/{split}/{plant}")
        
        shutil.move(image_path, f"{IMAGE_DIR}/{split}/{plant}")

        print(f"Moving image: {image_path}")

image_paths = test_data['image_path']

image_folders(image_paths, split = "test")

shutil.make_archive(f"{IMAGE_DIR}/test", 'zip', f"{IMAGE_DIR}/test")


dataset = load_dataset("imagefolder", data_dir = f"{IMAGE_DIR}")


train_dataset = dataset["train"]
test_dataset = dataset["test"]



def chunk_dataframe(df, chunk_size):
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    return chunks

train_chunks = chunk_dataframe(train_data, CHUNK_SIZE)


#test_data = pd.read_csv(f"{DIR}/data/test_data.csv")


#Preprocess image data
#---
#define function to process the images
def process_images(example):
    #define the current samples image path
    image_path = example['image_path']
    
    #open the current samples image
    image = Image.open(image_path).convert("RGB")
    
    #process the image for input into the ViT model
    pixel_values = processor(image, return_tensors = "pt")['pixel_values'].squeeze()
    example['pixel_values'] = pixel_values
    return example

#initialize the ViT model processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

for i in range(len(train_chunks)):
train_chunk = train_chunks[i]

#define the dataset object to contain the processed images
features = Features({
    'image_path': Value(dtype = 'string'),
    'label': ClassLabel(names = train_chunk['label'].unique().tolist())
})

#create the training dataset
train_dataset = Dataset.from_pandas(train_chunk, features = features)
train_dataset = train_dataset.map(process_images, batched = False)

#create the testing dataset
#test_dataset = Dataset.from_pandas(test_data, features = features)
#test_dataset = test_dataset.map(process_images, batched = False)

#save the datasets to disk
train_dataset.save_to_disk(f"{DIR}/data/train_dataset")
#test_dataset.save_to_disk(f"{DIR}/data/test_dataset")
