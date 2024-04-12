from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch

app = Flask(name)

# Load the diffusion model with torch.float32 data type
model_id = "nitrosocke/classic-anim-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Define the route to generate images based on prompts
@app.route('/generate_image', methods=['POST'])
def generate_image():
    # Get the prompt from the request
    prompt = request.json.get('prompt', '')

    # Generate image based on the prompt
    image = pipe(prompt).images[0]

    # Save the image
    image_path = f"./generated_image.png"
    image.save(image_path)

    # Return the path to the generated image
    return jsonify({'image_path': image_path})

if name == 'main':
    app.run(debug=True)