import streamlit as st
import os
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import uuid

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Define the layout of the app
st.set_page_config(page_title="NeuroSketch", page_icon=":medical_symbol:")

# Function to generate a unique filename for user input
def generate_user_input_filename():
    unique_id = uuid.uuid4().hex
    filename = f"user_input_{unique_id}.png"
    return filename

# Function to predict Parkinson's disease
def predict_parkinsons(img_path):
    best_model = load_model("./keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Get the numpy array (4-channel RGBA 100,100,4)
    input_numpy_array = np.array(img_path.image_data)

    # Get the RGBA PIL image
    input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")

    # Generate a unique filename for the user input
    user_input_filename = generate_user_input_filename()

    # Save the image with the generated filename
    input_image.save(user_input_filename)

    # Replace this with the path to your image
    image = Image.open(user_input_filename).convert("RGB")

    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = best_model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    Detection_Result = f"The model has detected {class_name[2:]}, with Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%."
    os.remove(user_input_filename)
    return Detection_Result, prediction

# Define the layout
st.title("NeuroSketch")
st.header("Unveiling Parkinson's with Precision")
st.write("Try drawing a Spiral and watch how an AI Model will detect the Parkinson Disease.")
st.warning("Warning: Do not click Submit Sketch button before drawing spiral on below Canvas.")

# Sidebar configurations
with st.sidebar:
    #img = Image.open("./Images/parkinson_disease_detection.jpg")
    #st.image(img)
    st.subheader("About NeuroSketch")
    link_text = "Distinguishing Different Stages of Parkinson’s Disease Using Composite Index of Speed and Pen-Pressure of Sketching a Spiral"
    link_url = "https://www.frontiersin.org/articles/10.3389/fneur.2017.00435/full"
    st.write("Parkinson's disease is a neurodegenerative disorder that affects motor functions, leading to tremors, stiffness, and impaired movement. The research presented in the article link mentioned below explores the use of spiral and wave sketch images to develop a robust algorithm for Parkinson's disease detection. NeuroSketch leverages these sketch images to train an AI model, achieving an impressive accuracy rate of 83%.")
    st.markdown(f"[{link_text}]({link_url})")

    st.header("Dataset")
    #img = Image.open("./Images/healthy_diseased_classification.jpeg")
    #st.image(img)

    st.header("Drawing Canvas Configurations")
    drawing_mode = "freedraw"
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.color_picker("Stroke color hex: ")
    bg_color = st.color_picker("Background color hex: ", "#eee")
    #bg_image = st.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.checkbox("Update in realtime", True)

# Create canvas component
canvas_size = 345
canvas_image = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    width=canvas_size,
    height=canvas_size,
    #background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Display preview of the sketch
st.subheader("Preview")
if canvas_image.image_data is not None:
    input_numpy_array = np.array(canvas_image.image_data)
    input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
    st.image(input_image, use_column_width=True)

# Predict Parkinson's disease on button click
submit = st.button(label="Submit Sketch")
if submit:
    st.subheader("Output")
    classified_label, prediction = predict_parkinsons(canvas_image)
    with st.spinner(text="This may take a moment..."):
        st.write(classified_label)

        # Display confidence scores for other classes
        class_names = open("labels.txt", "r").readlines()
        data = {"Class": class_names, "Confidence Score": prediction[0]}
        df = pd.DataFrame(data)
        df["Confidence Score"] = df["Confidence Score"].apply(lambda x: f"{str(np.round(x*100))[:-2]}%")
        df["Class"] = df["Class"].apply(lambda x: x.split(" ")[1])
        st.subheader("Confidence Scores on other classes:")
        st.write(df)



#python -m streamlit run streamlitApp.py
