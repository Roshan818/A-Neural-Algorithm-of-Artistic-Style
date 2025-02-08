import streamlit as st
from style_transfer import StyleTransfer
import torch
from PIL import Image
import io
import torchvision.transforms as transforms

st.set_page_config(page_title="Neural Style Transfer", page_icon="🎨")

st.title("Neural Style Transfer")
st.write("Upload a content image and a style image to create artistic compositions!")

# File uploaders
content_file = st.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
style_file = st.file_uploader("Choose a Style Image", type=["png", "jpg", "jpeg"])

# Sidebar controls
st.sidebar.title("Style Transfer Parameters")
num_steps = st.sidebar.slider("Number of Steps", 100, 500, 300)
style_weight = st.sidebar.slider("Style Weight", 1e4, 1e7, 1e6)
content_weight = st.sidebar.slider("Content Weight", 1, 1000, 1)

def process_image(image_file):
    if image_file is not None:
        # Save the uploaded file temporarily
        img_path = f"temp_{image_file.name}"
        with open(img_path, "wb") as f:
            f.write(image_file.getbuffer())
        return img_path
    return None

if content_file and style_file:
    if st.button("Generate Style Transfer"):
        with st.spinner("Applying style transfer... This may take a few minutes."):
            try:
                # Process uploaded files
                content_path = process_image(content_file)
                style_path = process_image(style_file)
                
                # Device configuration
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Initialize and run style transfer
                style_transfer = StyleTransfer(content_path, style_path, device)
                output = style_transfer.run_style_transfer(
                    num_steps=num_steps,
                    style_weight=style_weight,
                    content_weight=content_weight
                )
                
                # Convert output tensor to image
                output = output.cpu().squeeze(0)
                output = transforms.ToPILImage()(output)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Content Image")
                    st.image(content_file)
                with col2:
                    st.subheader("Style Image")
                    st.image(style_file)
                with col3:
                    st.subheader("Result")
                    st.image(output)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.markdown("""
### How to use:
1. Upload a content image (the base image you want to style)
2. Upload a style image (the image whose artistic style you want to apply)
3. Adjust the parameters in the sidebar if desired
4. Click 'Generate Style Transfer' and wait for the magic to happen!

### Parameters:
- **Number of Steps**: More steps generally means better results but longer processing time
- **Style Weight**: Higher values make the result more stylized
- **Content Weight**: Higher values make the result more similar to the content image
""")
