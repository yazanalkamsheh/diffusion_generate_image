import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load the diffusion model with torch.float32 data type
model_id = "nitrosocke/classic-anim-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Define the function to generate images based on prompts
def generate_image(prompt):
    # Generate image based on the prompt
    image = pipe(prompt).images[0]

    return image

# Streamlit app layout
st.title("Image Generation App")
prompt = st.text_input("Enter Prompt")
if st.button("Generate Image"):
    if prompt:
        image = generate_image(prompt)
        st.image(image, caption='Generated Image', use_column_width=True)

