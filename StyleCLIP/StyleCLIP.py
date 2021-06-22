import os
import argparse

import torch
import torchvision
import clip
import numpy as np
from PIL import Image

from StyleCLIP.stylegan_models import g_all, g_synthesis, g_mapping
from StyleCLIP.utils import GetFeatureMaps, transform_img, compute_loss

torch.manual_seed(20)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
vgg_layers = vgg16.features

vgg_layer_name_mapping = {
    '1': "relu1_1",
    '3': "relu1_2",
    '6': "relu2_1",
    '8': "relu2_2",
    # '15': "relu3_3",
    # '22': "relu4_3"
}

g_synthesis.eval()
g_synthesis.to(device)

class StyleCLIP(object):
    def __init__(self, output_path = './generations', prompt = ""):
    self.output_path = output_path
    self.batch_size = 1
    self.prompt = prompt
    self.lr = 1e-2
    self.img_save_freq = 10
    self.ref_img_path = ''

    self.output_dir = os.path.join(output_path, f'"{prompt}"')

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print("USING ", self.device)

    if not os.path.exists(self.output_dir):
        os.makedirs(self.output_dir)

    self.latent_shape = (batch_size, 1, 512)

    self.normal_generator = torch.distributions.normal.Normal(
        torch.tensor([0.0]),
        torch.tensor([1.]),
    )

    # init_latents = normal_generator.sample(latent_shape).squeeze(-1).to(device)
    self.latents_init = torch.zeros(latent_shape).squeeze(-1).to(device)
    self.latents = torch.nn.Parameter(latents_init, requires_grad=True)

    self.optimizer = torch.optim.Adam(
        params=[latents],
        lr=lr,
        betas=(0.9, 0.999),
    )

    self.clip_transform = torchvision.transforms.Compose([
    # clip_preprocess.transforms[2],
    clip_preprocess.transforms[4],
    ])

    if self.ref_img_path is None:
        self.ref_img = None
    else:
        self.ref_img = clip_preprocess(Image.open(ref_img_path)).unsqueeze(0).to(device)

    self.clip_normalize = torchvision.transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )


    def truncation(self, x, threshold=0.7, max_layer=8):
        avg_latent = torch.zeros(batch_size, x.size(1), 512).to(self.device)
        interp = torch.lerp(avg_latent, x, threshold)
        do_trunc = (torch.arange(x.size(1)) < max_layer).view(1, -1, 1).to(self.device)
        return torch.where(do_trunc, interp, x)

    def tensor_to_pil_img(self, img):
        img = (img.clamp(-1, 1) + 1) / 2.0
        img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        img = Image.fromarray(img.astype('uint8'))
        return img

    def compute_clip_loss(self, img, text):
        # img = clip_transform(img)
        img = torch.nn.functional.upsample_bilinear(img, (224, 224))
        tokenized_text = clip.tokenize([text]).to(device)

        img_logits, _text_logits = clip_model(img, tokenized_text)

        return 1/img_logits * 100

    def compute_perceptual_loss(self, gen_img, ref_img):
        gen_img = torch.nn.functional.upsample_bilinear(img, (224, 224))
        loss = 0
        len_vgg_layer_mappings = int(max(vgg_layer_name_mapping.keys()))

        ref_feats = ref_img
        gen_feats = gen_img

        for idx, (name, module) in enumerate(vgg_layers._modules.items()):
            ref_feats = module(ref_feats)
            gen_feats = module(gen_feats)
            if name in vgg_layer_name_mapping.keys():
                loss += torch.nn.functional.mse_loss(ref_feats, gen_feats)
            
            if idx >= len_vgg_layer_mappings:
                break
        
        return loss/len_vgg_layer_mappings

    def run(self):
        counter = 0
        path_to_res = ""
        while True:
            dlatents = latents.repeat(1,18,1)
            img = g_synthesis(dlatents)
            
            # NOTE: clip normalization did not seem to have much effect
            # img = clip_normalize(img)

            loss = self.compute_clip_loss(img, self.prompt)

            # NOTE: uncomment to use perceptual loos. Still WIP. You will need to define
            # the `ref_img_path` to use it. The image referenced will be the one 
            # used to condition the generation.
            # perceptual_loss = compute_perceptual_loss(img, ref_img)
            # loss = loss + perceptual_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if counter % self.img_save_freq == 0:
                img = tensor_to_pil_img(img)
                img.save(os.path.join(output_dir, f'{counter}.png'))
                path_to_res = os.path.join(output_dir, f'{counter}.png')

                print(f'Step {counter}')
                print(f'Loss {loss.data.cpu().numpy()[0][0]}')
                break

            counter += 1
        return path_to_res