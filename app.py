import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

model=load_model("my_model.h5")
st.title("MNIST Dataset Predictor")
img=st.camera_input("Camera")
def process(input_image):
    input_image=input_image.resize((28,28)).convert("L")
    input_image=np.array(input_image)
    input_image=input_image/255
    input_image=np.expand_dims(input_image,axis=0)
    input_image=input_image.reshape(1,28,28,1)
    return input_image
if img is not None:
    img=Image.open(img)
    image=process(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)
    class_names=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    st.write(class_names[predicted_class])
