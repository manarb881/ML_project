
Face Recognition App
This is a simple face recognition web application built using Streamlit, TensorFlow, and FAISS. The app allows users to upload a face image, which is then compared to a set of precomputed face embeddings to identify the closest match.

Features
Face Upload: Users can upload images of faces.
Face Recognition: The app uses a pre-trained FaceNet model to generate face embeddings and compares them using FAISS to find the closest match.
Similarity Search: FAISS is used to efficiently find similar embeddings in a large dataset.

Requirements
Before you run the app, ensure you have the following Python packages installed:

streamlit
Pillow
numpy
tensorflow
faiss-cpu (or faiss-gpu for GPU support)

