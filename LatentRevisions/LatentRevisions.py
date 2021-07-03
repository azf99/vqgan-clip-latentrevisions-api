from LatentRevisions.initialize import *
from LatentRevisions.utils import *

import numpy as np

#. A detailed, high-quality photo without distortions
text_other = '''incoherent, confusing, cropped, watermarks'''

class Pars(torch.nn.Module):
    def __init__(self, o_i2, sideX, sideY):
        super(Pars, self).__init__()
        self.sideX = sideX
        self.sideY = sideY
        # if optional_path_to_a_starter_image != '':
        self.normu = torch.nn.Parameter(o_i2.cuda().clone().view(1, 256, self.sideX//16 * self.sideY//16))
        # else:
        #   self.normu = .5*torch.randn(batch_size, 256, self.sideX//16 * self.sideY//16).cuda()
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
      return normu.clip(-6, 6).view(1, -1, self.sideX//16, self.sideX//16)

class LatentRevisions(object):
    def __init__(self, prompt):
        self.optional_path_to_a_starter_image = ''
        self.text_input = prompt
        self.w0 = 5 
        self.text_to_add = "" 
        self.w1 = -0.1 
        self.img_enc_path = ""
        self.w2 = 1.2 
        self.ne_img_enc_path = ""
        self.w3 = 0.3

        # How to weight the 2 texts (w0 and w1) and the images (w3 & w3)
        self.im_shape = [512, 512, 3]
        self.sideX, self.sideY, self.channels = self.im_shape
        self.batch_size = 1

        with torch.no_grad():
            if self.optional_path_to_a_starter_image != '':
              x = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(self.optional_path_to_a_starter_image)).unsqueeze(0).permute(0, 3, 1, 2)[:,:3], (self.sideX, self.sideY)) / 255).cuda()
            else:
              x = torch.nn.functional.interpolate(.5*torch.rand(size=(self.batch_size, 3, self.sideX//1, self.sideY//1)).cuda(), (self.sideX, self.sideY), mode='bilinear')
              x = kornia.augmentation.GaussianBlur((7, 7), (14, 14), p=1)(x)
              x = (x * 2 - 1)
              self.o_i1 = model16384.encoder(x)
              self.o_i2 = model16384.quant_conv(self.o_i1)
            # o_i3 = model16384.post_quant_conv(o_i2)

        #torch.cuda.empty_cache()
        self.dec = .1

        self.lats = Pars(self.o_i2, self.sideX, self.sideY).cuda()
        self.mapper = [self.lats.normu]
        self.optimizer = torch.optim.AdamW([{'params': self.mapper, 'lr': .5}], weight_decay=self.dec)
        self.eps = 0

        self.t = 0
        if self.text_input != '':
            tx = clip.tokenize(self.text_input)
            self.t = perceptor.encode_text(tx.cuda()).detach().clone()

        self.text_add = 0
        if self.text_to_add != '':
            self.text_add = clip.tokenize(text_to_add)
            self.text_add = perceptor.encode_text(self.text_add.cuda()).detach().clone()

        self.t_not = clip.tokenize(text_other)
        self.t_not = perceptor.encode_text(self.t_not.cuda()).detach().clone()


        self.nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        self.img_enc = 0
        if self.img_enc_path != '':
            self.img_enc = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(self.img_enc_path)).unsqueeze(0).permute(0, 3, 1, 2), (224, 224)) / 255).cuda()[:,:3]
            self.img_enc = self.nom(self.img_enc)
            self.img_enc = perceptor.encode_image(self.img_enc.cuda()).detach().clone()

        self.ne_img_enc = 0
        if self.ne_img_enc_path != '':
            self.ne_img_enc = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(self.ne_img_enc_path)).unsqueeze(0).permute(0, 3, 1, 2), (224, 224)) / 255).cuda()[:,:3]
            self.ne_img_enc = self.nom(self.ne_img_enc)
            self.ne_img_enc = perceptor.encode_image(self.ne_img_enc.cuda()).detach().clone()

        self.augs = torch.nn.Sequential(
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomAffine(24, (.1, .1)) # , fill=0)
        ).cuda()

        self.up_noise = .11
        self.itt = 0

        #with torch.no_grad():
        #al = (model(self.lats()).cpu().clip(-1, 1) + 1) / 2
        #for allls in al:
        #    displ(allls[:3])
        #    print('\n')

    def model(self, x):
        # o_i1 = model16384.encoder(x)
        # o_i1 = x
        # o_i2 = model16384.quant_conv(o_i1)
        self.o_i2 = x
        self.o_i3 = model16384.post_quant_conv(self.o_i2)
        i = model16384.decoder(self.o_i3)
        return i

    def update_area_and_prompt(self, latest):

        print('''Input a new text_input prompt
        or keep the same by leaving blank
        then press Enter.''')
        imeh = input()
        if imeh != '':
            tx = clip.tokenize(imeh)
            self.t = perceptor.encode_text(tx.cuda()).detach().clone()
        print('''Input the word "Draw" (or any other text) and press enter to go into revision mode to draw over which parts of the image should be optimized
        or leave this input blank to continue with the currently selected area, then press enter.''')
        inde = input()
        if inde != '':
            self.mapper = [self.lats.normu]
            optimizer = torch.optim.AdamW([{'params': self.mapper, 'lr': .5}], weight_decay=self.dec)
            
            img = np.array(latest)[0,:,:,:]
            img = np.transpose(img, (1, 2, 0))
            imageio.imwrite('/content/here.jpg', np.array(img))

            source = '/content/here.jpg'
            npth = '/usr/local/share/jupyter/nbextensions/latest.jpg'
            shutil.move(source, npth)

            self.lats.normu.data = self.lats.normu.scatter(2, self.lats.ignore.unsqueeze(0).unsqueeze(0).expand(-1, 256, -1), self.lats.keep.detach())

            _ = draw()

            drawn = torch.nn.functional.interpolate(torch.tensor(imageio.imread('/content/drawing.png')).unsqueeze(0).permute(0, 3, 1, 2), (self.sideX//16, self.sideY//16), mode='nearest')[:,3:4,:,:]

            ed = []
            zs = []
            for inx, kj in enumerate(drawn.view(-1, 1)):
              if kj.sum() < 1:
                  zs.append(inx)
              else:
                  ed.append(inx)

            self.lats.ignore = torch.tensor(zs).cuda()
            self.lats.keep = self.lats.normu[:, :, self.lats.ignore].detach()
            self.lats.keep_indices = torch.tensor(ed).cuda()
            if len(ed) > 0:
                self.lats.normu.data[:, :, torch.tensor(ed).cuda()] = torch.randn_like(self.lats.normu.data[:, :, torch.tensor(ed).cuda()])
        


    def augment(self, into, cutn=32):
        into = torch.nn.functional.pad(into, (self.sideX//2, self.sideX//2, self.sideX//2, self.sideX//2), mode='constant', value=0)
        into = self.augs(into)

        p_s = []
        for ch in range(cutn):
            # size = torch.randint(int(.5*self.sideX), int(1.9*self.sideX), ())
            size = int(torch.normal(1.2, .3, ()).clip(.43, 1.9) * self.sideX)
            
            if ch > cutn - 4:
              size = int(self.sideX*1.4)
              offsetx = torch.randint(0, int(self.sideX*2 - size), ())
              offsety = torch.randint(0, int(self.sideX*2 - size), ())
              apper = into[:, :, offsetx:offsetx + size, offsety:offsety + size]
              apper = torch.nn.functional.interpolate(apper, (int(224*scaler), int(224*scaler)), mode='bilinear', align_corners=True)
              p_s.append(apper)
        into = torch.cat(p_s, 0)

        into = into + self.up_noise*torch.rand((into.shape[0], 1, 1, 1)).cuda()*torch.randn_like(into, requires_grad=False)
        return into

    def checkin(self):
        #output.eval_js('new Audio("https://freesound.org/data/previews/80/80921_1022651-lq.ogg").play()')
        with torch.no_grad():
            print('''
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            ''')
            alnot = (self.model(self.lats()).cpu().clip(-1, 1) + 1) / 2
            for allls in alnot.cpu():
              displ(allls) #[:, int(.1*self.sideY):int(.9*self.sideY), int(.1*self.sideX):int(.9*self.sideX)]
              display.display(display.Image(str(3)+'.png'))
              print('\n')
        print('''
        ##########################################################
        ''', self.itt)
        #update_area_and_prompt(alnot)

        with torch.no_grad():
            alnot = (self.model(self.lats()).cpu().clip(-1, 1) + 1) / 2
            
            for allls in alnot.cpu():
              displ(allls) #[:, int(.1*self.sideY):int(.9*self.sideY), int(.1*self.sideX):int(.9*self.sideX)]
              display.display(display.Image(str(3)+'.png'))
              print('\n')

    def ascend_txt(self):
        out = self.model(self.lats())
        into = self.augment((out.clip(-1, 1) + 1) / 2)
        into = self.nom(into)
        iii = perceptor.encode_image(into)
        q = self.w0*self.t + self.w1*self.text_add + self.w2*self.img_enc + self.w3*self.ne_img_enc
        q = q / q.norm(dim=-1, keepdim=True)
        all_s = torch.cosine_similarity(q, iii, -1)
        # all_s = torch.arccos(0 - all_s) / np.pi
        return [0, -10*all_s + 5 * torch.cosine_similarity(self.t_not, iii, -1)]
        
    def train(self, i):
        if self.itt % 100 == 0:
            self.checkin()
        loss1 = self.ascend_txt()
        loss = loss1[0] + loss1[1]
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.itt % 100 == 0:
            print(loss1)
            print('up_noise', self.up_noise)
            for g in self.optimizer.param_groups:
              print(g['lr'], 'lr', g['weight_decay'], 'decay')

        # if itt > 400:
        #   for g in optimizer.param_groups:
        #     g['lr'] *= .995
        #     g['lr'] = max(g['lr'], .1)
        #   dec *= .995

        if self.lats.keep_indices.size()[0] != 0:
            if torch.abs(self.lats().view(batch_size, 256, -1)[:, :, self.lats.keep_indices]).max() > 5:
              for g in self.optimizer.param_groups:
                g['weight_decay'] = self.dec
            else:
              for g in self.optimizer.param_groups:
                g['weight_decay'] = 0        

    def run():
        while self.itt <= 100:
            train(self.itt)
            self.itt += 1