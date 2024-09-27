#!/usr/bin/python3

#Model testing
#---
#Script to test the fine-tuned ViT model.


#Table of contents:
#---
#1. Test the model accuracy
#2. Visualize the training history


#load libraries
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from datasets import load_from_disk
from transformers import ViTImageProcessor, AutoModelForImageClassification

#define global variables
DIR = "/mnt/c/Users/dstra/Nextcloud/Documents/Computers/Code/Github repos/Deep learning/BertaBloomID" #working directory
MODEL_DIR = "/mnt/g/Machine_learning/BertaBloomID" #model directory

#load data
logs_df = pd.read_csv(f"{MODEL_DIR}/data/trainer_history.csv") #training history
test_dataset = load_from_disk(f"{MODEL_DIR}/data/test_dataset") #test dataset


#1. Test the model accuracy
#---
#load the base and fine-tuned ViT models
models = {
    'base-model': AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        ignore_mismatched_sizes = True,
        device_map = 'cuda'),
    'fine-tuned model': AutoModelForImageClassification.from_pretrained(
        f"{MODEL_DIR}/model/BertaBloomID_model",
        device_map = 'cuda')
    }

#initialize the processor for both models
processor = ViTImageProcessor.from_pretrained(f"{MODEL_DIR}/model/BertaBloomID_processor")

#define function to return predictions and labels from each test example
def get_model_predictions(model, test_dataset, processor):
    #set the model to evaluate mode
    model.eval()
    
    #define empty lists
    predictions = []
    references = []
    for example in test_dataset:        
        #define the current examples input data and convert to tensors
        inputs = torch.tensor(example['pixel_values']).unsqueeze(0)
        inputs = inputs.to(model.device)
        
        #perform forward propagation
        with torch.no_grad():
            outputs = model(inputs)
        
        #get the predicted class (index with the highest logit value)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim = 1).item()
        
        #append the predicted class and class label
        predictions.append(predicted_class)
        references.append(example['label'])
    return predictions, references

#define empty dictionaries
accuracies = {}
cm_data = {}
#calculate the model accuracies
for model_name in models.keys():
    #define the current model
    model = models.get(model_name)
    
    #obtain the current models predicted classes and labels
    predictions, references = get_model_predictions(model, test_dataset, processor)
    
    #calculate the accuracy
    accuracy = accuracy_score(references, predictions)
    
    #calculate the confusion matrix
    cm = confusion_matrix(references, predictions)
    
    #store the results
    accuracies[model_name] = accuracy
    cm_data[model_name] = cm


#2. Visualize the training history
#---
#create the lineplot
plt.figure(figsize = (8, 6))
plt.plot(logs_df['epoch'], logs_df['train_loss'], label = 'Train Loss')
plt.plot(logs_df['epoch'], logs_df['eval_loss'], label = 'Eval Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f"Base model accuracy: {accuracies['base-model'] * 100:.2f} % | Fine-tuned model accuracy: {accuracies['fine-tuned model'] * 100:.2f} %")
plt.savefig(f"{DIR}/visualizations/training_history.jpg")
plt.close()
