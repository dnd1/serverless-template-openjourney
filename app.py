from io import BytesIO
from potassium import Potassium, Request, Response

from transformers import pipeline
import torch
import base64
from diffusers import StableDiffusionPipeline

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    device = 0 if torch.cuda.is_available() else -1
    # model = pipeline('fill-mask', model='bert-base-uncased', device=device)
    model_id = "prompthero/openjourney"
    # model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model = StableDiffusionPipeline.from_pretrained(
        model_id
    ).to("cuda")

    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    # prompt = request.json.get("prompt")
    model = context.get("model")
    # image = model(prompt).images[0]


    # return Response(
    #     # json = {"outputs": outputs[0]}, 
    #     json = {"outputs": 'image'}, 
    #     status=200
    # )
      # Parse out your arguments
    prompt = request.json.get('prompt')
    negative = request.json.get('negative')
    num_inference_steps = request.json.get('num_inference_steps', 50)
    guidance_scale = request.json.get('guidance_scale', 7)
    
    if prompt == None:
        return Response(json = {'message': "No prompt provided"})
    
    # Run the model
    # t1 = time.time()
    with torch.autocast_decrement_nesting():
        image = model(prompt, negative_prompt=negative, num_images_per_prompt=1, num_inference_steps=50, guidance_scale=7.5).images[0]

    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return Response(json = {'image_base64': image_base64}, status=200)

if __name__ == "__main__":
    app.serve()