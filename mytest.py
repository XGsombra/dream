import requests
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from nerf.nerf_clip import CLIP

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

guidance = CLIP("cpu")
image_z = guidance.get_image_embeds(image)
print(image_z.size())