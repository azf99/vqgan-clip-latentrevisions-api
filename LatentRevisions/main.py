from imports import *
from initialize import *
from utils import *

optional_path_to_a_starter_image = ''


text_input = 'a beautiful person'
w0 = 5 



text_to_add = "" 
w1 = -0.1 
img_enc_path = ""
w2 = 1.2 
ne_img_enc_path = ""
w3 = 0.3

# How to weight the 2 texts (w0 and w1) and the images (w3 & w3)
im_shape = [512, 512, 3]
sideX, sideY, channels = im_shape
batch_size = 1

with torch.no_grad():
    if optional_path_to_a_starter_image != '':
      x = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(optional_path_to_a_starter_image)).unsqueeze(0).permute(0, 3, 1, 2)[:,:3], (sideX, sideY)) / 255).cuda()
    else:
      x = torch.nn.functional.interpolate(.5*torch.rand(size=(batch_size, 3, sideX//1, sideY//1)).cuda(), (sideX, sideY), mode='bilinear')
      x = kornia.augmentation.GaussianBlur((7, 7), (14, 14), p=1)(x)
    x = (x * 2 - 1)
    o_i1 = model16384.encoder(x)
    o_i2 = model16384.quant_conv(o_i1)
    # o_i3 = model16384.post_quant_conv(o_i2)

torch.cuda.empty_cache()





#. A detailed, high-quality photo without distortions



text_other = '''incoherent, confusing, cropped, watermarks'''


class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()


        # if optional_path_to_a_starter_image != '':
        self.normu = torch.nn.Parameter(o_i2.cuda().clone().view(batch_size, 256, sideX//16 * sideY//16))
        
        # else:
        #   self.normu = .5*torch.randn(batch_size, 256, sideX//16 * sideY//16).cuda()
          
        #   self.normu = torch.nn.Parameter(torch.sinh(1.9*torch.arcsinh(self.normu)))

        self.ignore = torch.empty(0,).long().cuda()

        self.keep = torch.empty(0,).long().cuda()

        self.keep_indices = torch.empty(0,).long().cuda()

    def forward(self):


      # can't remember if this is necessary lmao
      mask = torch.ones(self.normu.shape, requires_grad=False).cuda()
      mask[:, :, self.ignore] = 1
      normu = self.normu * mask

      normu.scatter_(2, self.ignore.unsqueeze(0).unsqueeze(0).expand(-1, 256, -1), self.keep.detach())


      return normu.clip(-6, 6).view(1, -1, sideX//16, sideX//16)
      

def model(x):
  # o_i1 = model16384.encoder(x)
  # o_i1 = x
  # o_i2 = model16384.quant_conv(o_i1)
  o_i2 = x
  o_i3 = model16384.post_quant_conv(o_i2)
  i = model16384.decoder(o_i3)
  return i


dec = .1

lats = Pars().cuda()
mapper = [lats.normu]
optimizer = torch.optim.AdamW([{'params': mapper, 'lr': .5}], weight_decay=dec)
eps = 0

t = 0
if text_input != '':
  tx = clip.tokenize(text_input)
  t = perceptor.encode_text(tx.cuda()).detach().clone()

text_add = 0
if text_to_add != '':
  text_add = clip.tokenize(text_to_add)
  text_add = perceptor.encode_text(text_add.cuda()).detach().clone()

t_not = clip.tokenize(text_other)
t_not = perceptor.encode_text(t_not.cuda()).detach().clone()


nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

img_enc = 0
if img_enc_path != '':
  img_enc = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(img_enc_path)).unsqueeze(0).permute(0, 3, 1, 2), (224, 224)) / 255).cuda()[:,:3]
  img_enc = nom(img_enc)
  img_enc = perceptor.encode_image(img_enc.cuda()).detach().clone()

ne_img_enc = 0
if ne_img_enc_path != '':
  ne_img_enc = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(ne_img_enc_path)).unsqueeze(0).permute(0, 3, 1, 2), (224, 224)) / 255).cuda()[:,:3]
  ne_img_enc = nom(ne_img_enc)
  ne_img_enc = perceptor.encode_image(ne_img_enc.cuda()).detach().clone()



augs = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomAffine(24, (.1, .1), fill=0)
).cuda()


up_noise = .11



itt = 0


with torch.no_grad():
  al = (model(lats()).cpu().clip(-1, 1) + 1) / 2
  for allls in al:
    displ(allls[:3])
    print('\n')