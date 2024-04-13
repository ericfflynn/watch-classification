from fastai.vision.all import load_learner
import streamlit as st
from PIL import Image

st.header("Rolex Classification Model")
st.subheader("Upload an image of a Datejust, Daytona, Gmt-Master, or Submariner below and see if the model can guess correctly!")
cat = ('Datejust', 'Daytona', 'Gmt-Master', 'Submariner')
learn = load_learner('rolexes.pkl')

def convert_to_percentage(dictionary):
    return {key: f"{round(value * 100, 2):.2f}%" for key, value in dictionary.items()}

def classify_watch(img):
    pred,idk,probs = learn.predict(img)
    watch_probs =  dict(zip(cat, map(float,probs)))
    return watch_probs, pred


img_file_buffer = st.file_uploader("Some images may fail upload. Please try again with a different image")
if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert('RGB')
    st.image(image)
    st.write("The prediction is: ", classify_watch(image)[1])
    st.write(convert_to_percentage(classify_watch(image)[0]))