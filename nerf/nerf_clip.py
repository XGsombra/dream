import torch
import torch.nn as nn

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import clip

class CLIP(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device, jit=False)
        
         # image augmentation
        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # self.gaussian_blur = T.GaussianBlur(15, sigma=(0.1, 10))

    
    def get_text_embeds(self, prompt, negative_prompt):

        # NOTE: negative_prompt is ignored for CLIP.

        text = clip.tokenize(prompt).to(self.device)
        text_z = self.clip_model.encode_text(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)

        return text_z

    def get_text_diff(self, text, dir_text):
        text_input = clip.tokenize(text).to(self.device)
        dir_text_input = clip.tokenize(dir_text).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_input)
            dir_text_embeddings = self.clip_model.encode_text(dir_text_input)
        return dir_text_embeddings - text_embeddings
    
    def get_image_embeds(self, image, negative_prompt, dir_diff=None):

        # NOTE: negative_prompt is ignored for CLIP.

        image_z = self.clip_model.encode_image(image)
        if dir_diff is not None:
            image_z = image_z + dir_diff
        image_z = image_z / image_z.norm(dim=-1, keepdim=True)

        return image_z

    
    def train_step(self, image_z, pred_rgb):

        pred_rgb = self.aug(pred_rgb)

        pred_image_z = self.clip_model.encode_image(pred_rgb)
        pred_image_z = pred_image_z / pred_image_z.norm(dim=-1, keepdim=True) # normalize features
        loss = - (pred_image_z * image_z).sum(-1).mean()

        return loss

