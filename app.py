import streamlit as st
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
import os
import tempfile
from style_transfer import StyleTransfer

st.set_page_config(page_title="Neural Style Transfer", page_icon="🎨", layout="wide")

# Initialize session state
if 'progress_images' not in st.session_state:
    st.session_state.progress_images = []

def update_progress(progress_data):
    st.session_state.progress_images.append(progress_data)

st.title("Neural Style Transfer")
st.write("Upload a content image and a style image to create artistic compositions!")

col1, col2 = st.columns(2)
with col1:
    content_file = st.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
with col2:
    style_file = st.file_uploader("Choose a Style Image", type=["png", "jpg", "jpeg"])

# Sidebar controls
st.sidebar.title("Style Transfer Parameters")
num_steps = st.sidebar.slider("Number of Steps", 100, 500, 300)
style_weight = st.sidebar.slider("Style Weight", 1e4, 1e7, 1e6)
content_weight = st.sidebar.slider("Content Weight", 1, 1000, 1)
show_intermediate = st.sidebar.checkbox("Show intermediate results", True)

def process_image(image_file):
    if image_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image_file.getbuffer())
            return tmp.name
    return None

if content_file and style_file:
    if st.button("Generate Style Transfer"):
        try:
            # Process uploaded files
            content_path = process_image(content_file)
            style_path = process_image(style_file)
            
            # Device configuration
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Clear previous progress
            st.session_state.progress_images = []
            
            # Progress placeholder
            progress_placeholder = st.empty()
            latest_result = st.empty()
            metrics_placeholder = st.empty()
            
            # Initialize and run style transfer
            style_transfer = StyleTransfer(
                content_path, 
                style_path, 
                device,
                intermediate_callback=update_progress if show_intermediate else None
            )
            
            output, progress_images, best_image = style_transfer.run_style_transfer(
                num_steps=num_steps,
                style_weight=style_weight,
                content_weight=content_weight
            )
            
            # Display final results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Content Image")
                st.image(content_file)
            with col2:
                st.subheader("Style Image")
                st.image(style_file)
            with col3:
                st.subheader("Final Result")
                st.image(style_transfer.tensor_to_pil(best_image))
            
            # Display progress gallery
            if show_intermediate and progress_images:
                st.subheader("Progress Gallery")
                gallery = st.columns(4)
                for idx, progress in enumerate(progress_images):
                    if idx % 4 == 0:
                        gallery = st.columns(4)
                    with gallery[idx % 4]:
                        st.image(progress['image'], caption=f"Step {progress['step']}")
                        st.write(f"Loss: {progress['loss']:.2f}")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.markdown("""
### Tips for better results:
- Start with a higher style weight and gradually decrease if needed
- Increase the number of steps for better quality
- Use the intermediate results to understand the optimization process
""")
