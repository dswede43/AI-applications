#App UI - BertaBloomID
#---
#Scipt to run the app UI via streamlit.


#import libraries
import helpers
import streamlit as st
import pandas as pd
from PIL import Image

#define global variables
MODEL_DIR = "./model"

#load data
metadata = pd.read_csv("./data/metadata.csv")

#load the fine-tuned ViT model
model, processor = helpers.load_model(MODEL_DIR)

#upload the image
uploaded_image = st.file_uploader("Upload your plant image...")

#if a file is uploaded
if uploaded_image is not None:
    #open the image
    image = Image.open(uploaded_image)

    #display the image
    st.image(image, use_column_width = True)

    #convert and resize the image
    image = image.convert("RGB")
    image = image.resize((224, 224))

    if st.button('Identify this plant!'):
        #return the predicted plant species
        predicted_class, confidence_score = helpers.classify_image(image, model, processor)

        #obtain the predicted class species name
        species_name = metadata[metadata['label'] == predicted_class]['species_name']
        species_name = species_name.to_string(index = False)

        #display the plant species name
        st.write("Plant species name: ", species_name)
        st.write("Prediction confidence: ", f"{round(confidence_score, 2) * 100}%")

        #provide warning message for low confidence prediction
        if (confidence_score < 0.5):
            st.warning("The confidence in this plant classification is low.")

#make a UI to show the results of some test images (use own images)
