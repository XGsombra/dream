import requests
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModelWithProjection

from nerf.nerf_clip import CLIP
from sd import StableDiffusion

def clip_normalize(image):

    image = F.interpolate(image, size=224, mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711])
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    image = (image - mean) / std

    return image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
convert_tensor = transforms.ToTensor()
image = convert_tensor(image)
image = image.unsqueeze(0)
print(image.size())
image = clip_normalize(image)

# guidance = CLIP("cpu")
# image_z = guidance.get_image_embeds(image)
# print(image_z.size())
# guidance = StableDiffusion("cpu", "2.1", None)
image = image.to("cuda")
model_key = "stabilityai/stable-diffusion-2-1-unclip"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_key, subfolder="image_encoder").to("cuda")
tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer").to("cuda")
text_encoder = CLIPTextModelWithProjection.from_pretrained(model_key, subfolder="text_encoder").to("cuda")
# print(image_encoder(image).image_embeds.size())

###########################################################
with torch.no_grad():
    text_embeddings = text_encoder("asdasdasdada".input_ids.to("cuda"))[0]

    image_embeddings = image_encoder(image).image_embeds.unsqueeze(0)
    print(image_embeddings.size())


    # Do the same for unconditional embeddings
    uncond_input = tokenizer("negative_prompt", padding='max_length', max_length=tokenizer.model_max_length,
                              return_tensors='pt')

with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0]

    print(uncond_embeddings.size())

# Cat for final embeddings
image_embeddings = torch.cat([uncond_embeddings, image_embeddings])
print(image_embeddings.size())