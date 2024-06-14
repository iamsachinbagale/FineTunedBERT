import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
import base64
import matplotlib.pyplot as plt

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('Image.jpg')

plt.style.use("ggplot")

st.title("Toxic Comment Classification")

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("Sachin101/FineTunedBert")
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
    
  1:'Toxic',
  0:'Non Toxic'
}

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])
