import gradio as gr
import torch
from torchvision import transforms
from PIL import Image


# Function to translate the image
def translate_image(image):
    return image

# Set up the Gradio interface
interface = gr.Interface(
    fn=translate_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil", label="Translated Image"),
    title="Correction App",
    description="Upload an image and get the translated version."
)
# Launch the Gradio app
interface.launch()
