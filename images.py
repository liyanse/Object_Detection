import streamlit as st
import subprocess


def label_images(image_path):
    subprocess.run(["labelImg", image_path])

def upload_images():
    uploaded_files = st.file_uploader("Upload some images", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        st.image(uploaded_file)
        
def get_labels():
    label_input = st.text_input("Enter new class labels (comma separated)")
    return label_input.split(",")

def update_model(images, labels):
    # Add images and labels to the training dataset
    # Retrain the YOLOV5 model with the updated dataset
    # Save the new weights to a file
    pass

if st.button("Add new images"):
    image_path = st.text_input("Enter image directory path")
    label_images(image_path)

if st.button("Upload new images"):
    upload_images()

if st.button("Add new class labels"):
    labels = get_labels()

if st.button("Update model"):
    update_model(images, labels)
    st.success("Model updated successfully!")








