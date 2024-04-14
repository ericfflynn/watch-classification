import streamlit as st
from PIL import Image
import numpy as np
from fastai.vision.all import *

learn = load_learner('rolexes.pkl')

watch_categories = ['Datejust', 'Daytona', 'Gmt-Master', 'Submariner']

def predict(image):
    img = PILImage.create(image).convert('RGB')
    pred, pred_idx, probs = learn.predict(img)
    return pred, probs

def display_classification_results(prediction):
    fig, ax = plt.subplots()
    ax.bar(watch_categories, prediction)
    ax.set_ylabel('Probability')
    st.pyplot(fig)


def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: black;'>Rolex Watch Classifier</h1>", unsafe_allow_html=True)
    st.markdown("""
    <h3 style='text-align: center; color: black;'>Feel free to play around with the different example photos and then try to upload your own. Even if the watch is not one of these four models, it will tell you which model is most similar!</h1>
    <p style='text-align: center; color: black;'>Note: Certain mages may fail to upload please try again with a new image</p>
    """,
    unsafe_allow_html=True)
    st.write('***')
    st.sidebar.title('Image Selection')
    uploaded_file = st.sidebar.file_uploader("Upload your own image:", type=['jpg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    else:
        default_images = [
        'datejust.jpg',
        'submariner.jpg',
        'gmt-master.jpg',
        'daytona.jpg',
        ]
        selected_image = st.sidebar.radio("Select a default image:", default_images,format_func=lambda x: x.split('.')[0].title())
        image = Image.open(selected_image).resize((600, 400))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h2 style='text-align: center; color: black;'>Image In Question</h1>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        
    with col2:
        st.markdown(f"<h2 style='text-align: center; color: black;'>Classification Probabilities</h1>", unsafe_allow_html=True)
        prediction, probs = predict(image)
        display_classification_results(probs)
        st.markdown(f"<h3 style='text-align: center; color: black;'>Prediction: {prediction.title()}</h1>", unsafe_allow_html=True)
    st.write('***')
    st.markdown(f"<p style='text-align: center; color: black;'>You can find the notebook used to generate this classification model <a href='https://www.kaggle.com/code/ericfflynn/rolex-classification-model/notebook'>here</a>.</h1>", unsafe_allow_html=True)
    


if __name__ == '__main__':
    main()
