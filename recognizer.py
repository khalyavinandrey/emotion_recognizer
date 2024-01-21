import streamlit as st
import tensorflow as tf
import numpy as np


def predict_model(prediction_image):
    model = tf.keras.models.load_model("my_keras_model.h5")
    image = tf.keras.preprocessing.image.load_img(prediction_image, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


st.header("Determine a person's emotions from a photo:")
uploaded_image = st.file_uploader("Select a person's photo:")
if uploaded_image is not None:
    st.image(uploaded_image, width=8, use_column_width=True)
    if st.button("Predict"):
        with st.spinner("Wait"):
            result_index = predict_model(uploaded_image)
            with open("emotions.txt", encoding='utf-8') as f:
                content = f.readlines()
            label = []
        for i in content:
            label.append(i[:-1])
        st.success("The person is {}".format(label[result_index]))
