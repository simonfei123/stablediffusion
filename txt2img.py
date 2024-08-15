import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# prompt = "a symbolic drawing of a acorn: logo-like, black-and-white, no words, flat"
# prompt = "a simple drawing of a acorn: symbolic, logo-like, black-and-white, no words, flat"
# prompt = "a conceptual drawing of a abacus: symbolic, logo-like, black-and-white, no words, flat"
# prompt = "a conceptual drawing of a abacus: simple, cartoonish, black-and-white, flat"
prompt = "a contour drawing of a aardvark: simple, cartoonish, black-and-white, flat"
# prompt = "a contour drawing of a air_conditioner: simple, cartoonish, black-and-white, flat, white background"
# prompt = "an emoji of a acorn: symbolic, logo-like, black-and-white, no words, flat"
# negative_prompt = 'off-centered, small, complicated, having words, photorealism, cropped, detailed, abstract'
# negative_prompt = ''
# negative_prompt = 'off-centered, small, complicated, having words, photorealism, cropped, detailed, abstract, complex patterns, woodcut'
negative_prompt = 'off-centered, having words, cropped, woodcut'
torch.manual_seed(0)
image = pipe(prompt, negative_prompt=negative_prompt).images[0]
    
image.save("output.png")