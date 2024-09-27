#!/usr/bin/python3

#Model fine-tuning
#---
#Script to fine-tune the ViT model.


#load libraries
import torch
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_from_disk
from transformers import ViTImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer

#define global variables
#DIR = "/mnt/c/Users/dstra/Nextcloud/Documents/Computers/Code/Github repos/Deep learning/BertaBloomID" #working directory
DIR = "/mnt/g/Machine_learning/BertaBloomID"
#IMAGE_DIR = "/mnt/g/Machine_learning/BertaBloomID_images" #direcory to download images

#load data
train_dataset = load_from_disk(f"{DIR}/data/train_dataset")
test_dataset = load_from_disk(f"{DIR}/data/test_dataset")
#dataset = load_dataset("imagefolder", data_dir = f"{IMAGE_DIR}")
#train_dataset = dataset["train"]
#test_dataset = dataset["test"]


#ViT model training
#---
#initialize the ViT model processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

#initialize the ViT model
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    #num_labels = len(train_data['label'].unique()),
    ignore_mismatched_sizes = True)

#define training arguments
training_args = TrainingArguments(
    output_dir = f"{DIR}/model",
    logging_dir = f"{DIR}/logs",
    eval_strategy = "epoch",
    logging_strategy = "epoch",
    per_device_train_batch_size = 20,
    per_device_eval_batch_size = 20,
    num_train_epochs = 10,
    save_strategy = "epoch",
    save_total_limit = 2,
    remove_unused_columns = False,
)

#initialize the trainer
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    tokenizer = processor,
)

#train the model
trainer.train()

#save the fine-tuned model
model.save_pretrained(f"{DIR}/model/BertaBloomID_model")
processor.save_pretrained(f"{DIR}/model/BertaBloomID_processor")

#save the training history data
logs = pd.DataFrame(trainer.state.log_history)
epoch = pd.Series(logs['epoch'].dropna().unique(), name = 'epoch').reset_index(drop = True)
train_loss = pd.Series(logs['loss'].dropna(), name = 'train_loss').reset_index(drop = True)
eval_loss = pd.Series(logs['eval_loss'].dropna(), name = 'eval_loss').reset_index(drop = True)

#convert to data frame
logs_df = pd.concat([epoch, train_loss, eval_loss], axis = 1)

#save results as CSV
logs_df.to_csv(f"{DIR}/data/trainer_history.csv")
