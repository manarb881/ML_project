import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import faiss

# Load FaceNet model from TensorFlow Hub
facenet_model = hub.load('https://tfhub.dev/google/facenet/1')

# Set up custom CSS for better UI
st.markdown(
    """
    <style>
    .title {
        font-size: 50px;
        color: #3498db;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .upload-box {
        border: 2px dashed #3498db;
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f4f8;
    }
    .match-label {
        font-size: 30px;
        color: #2ecc71;
        font-weight: bold;
    }
    .image-container {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with custom CSS
st.markdown('<div class="title">Face Recognition App</div>', unsafe_allow_html=True)

# Sidebar for image upload and app info
st.sidebar.title("About the App")
st.sidebar.write("""
This is a face recognition app using the FaceNet model and FAISS for similarity search. Upload an image, and the app will find the closest match from the stored face embeddings.
""")

# Upload box styling
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# Function to preprocess images for FaceNet
def preprocess_image(image):
    image = image.resize((160, 160))  # Resize to 160x160
    image_array = np.asarray(image).astype('float32') / 255.0  # Normalize pixel values to [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

dataset_path = 'C:\Users\Admin\Downloads\archive\lfw-deepfunneled\lfw-deepfunneled'
embeddings = [] 
labels = [] 
for filename in os.listdir(dataset_path): 
    if filename.endswith('.jpg') or filename.endswith('.png'): 
        mage_path = os.path.join(dataset_path, filename)
        image_array = preprocess_image(image_path)
        embedding = facenet_model(image_array)
        embeddings.append(embedding.numpy()[0])
        label = os.path.splitext(filename)[0]
        labels.append(label)

embeddings_array = np.array(embeddings)
labels_array = np.array(labels)


np.save('stored_embeddings.npy', embeddings_array)
np.save('stored_labels.npy', labels_array)

# Function to generate embedding for a new face image
def generate_embedding(image_array):
    embedding = facenet_model(image_array)
    return embedding.numpy()[0]  # Return embedding as numpy array

# Function to search for the most similar face in the FAISS index
def find_similar_face(embedding):
    stored_embeddings = np.load('stored_embeddings.npy')  # Load stored embeddings
    index = faiss.IndexFlatL2(128)  # Create a FAISS index for 128-dimensional embeddings
    index.add(stored_embeddings)  # Add embeddings to the index
    _, closest_index = index.search(np.array([embedding]), 1)  # Search for the closest match
    return closest_index[0][0]  # Return the index of the closest match

# Function to get label (name) of the closest match
def get_label(index):
    labels = np.load('stored_labels.npy')  # Load labels corresponding to stored embeddings
    return labels[index]  # Return label of the closest match

if uploaded_file is not None:
    # Display uploaded image with style
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess the uploaded image
    image_array = preprocess_image(image)

    # Display a progress spinner while processing
    with st.spinner('Processing image...'):
        # Generate embedding for the uploaded image
        embedding = generate_embedding(image_array)

        # Find the closest match
        closest_index = find_similar_face(embedding)

        # Get the label (name) of the closest match
        label = get_label(closest_index)

    # Display the result with custom styling
    st.markdown(f'<div class="match-label">Closest match: {label}</div>', unsafe_allow_html=True)

