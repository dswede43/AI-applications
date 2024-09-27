#Helper functions - BertaBloomID
#---
#Helper functions for the web application GUI.


#Functions
#---
#1. Function to load the model and its processor
#2. Function to return the models predicted class and confidence


#import libraries
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, AutoModelForImageClassification


#1. Function to load the model and its processor
#---
def load_model(dir):
    #load fine-tuned ViT model
    model = AutoModelForImageClassification.from_pretrained(
        f"{dir}/BertaBloomID_model",
        device_map = 'cuda')

    #load the model processor
    processor = ViTImageProcessor.from_pretrained(f"{dir}/BertaBloomID_processor")

    return model, processor


#2. Function to return the models predicted class and confidence
#---
def classify_image(image, model, processor):
    #define the current examples input data and convert to tensors
    inputs = processor(image, return_tensors = "pt")['pixel_values']
    inputs = inputs.to(model.device)

    #perform forward propagation
    with torch.no_grad():
        outputs = model(inputs)

    #get the model logits (raw output)
    logits = outputs.logits

    #obtain the model predicted class (index with the highest logit value)
    predicted_class = torch.argmax(logits, dim = 1).item()

    #obtain the models confidence in the predicted class (max probability)
    probs = F.softmax(logits, dim = 1)
    confidence_score = torch.max(probs).item()    

    return predicted_class, confidence_score
