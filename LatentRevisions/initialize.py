from LatentRevisions.imports import *
import clip
perceptor, preprocess = clip.load('ViT-B/32', jit=False)
perceptor.eval().requires_grad_(False)

clip.available_models()

perceptor.visual.input_resolution

scaler = 1

import sys
sys.path.append(".")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.chdir("taming-transformers")

import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None):
  model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

os.chdir("../")

config16384 = load_config("./logs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)
model16384 = load_vqgan(config16384, ckpt_path="./logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(DEVICE).eval().requires_grad_(False);