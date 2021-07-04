# NOTE: Still for testing on Colab
# TODO: Add all python dependencies in requirements.txt

# download a VQGAN with a larger codebook (16384 entries)
echo "Downloading and saving VQGAN weights...."
mkdir -p "./logs/vqgan_imagenet_f16_16384/checkpoints"
mkdir -p "./logs/vqgan_imagenet_f16_16384/configs"

if [ -z "$(ls -A ./logs/vqgan_imagenet_f16_16384/checkpoints/)" ]; then
  wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O './logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt' 
  wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O './logs/vqgan_imagenet_f16_16384/configs/model.yaml' 
fi
echo "VQGAN weights saved...."

wget https://github.com/lernapparat/lernapparat/releases/download/v2019-02-01/karras2019stylegan-ffhq-1024x1024.for_g_all.pt

sudo apt-get install redis

pip3 install --no-deps ftfy regex tqdm redis flask flask_cors
pip3 install kornia==0.5.4
pip3 install git+https://github.com/openai/CLIP.git

pip3 uninstall torchtext --yes
pip3 install einops

#cd /content/
git clone https://github.com/CompVis/taming-transformers  
cd taming-transformers

# !cp /content/drive/MyDrive/vqgan_imagenet_f16_16384-20210325T002625Z-001.zip /content/vq.zip
# !unzip /content/vq.zip -d /content/taming-transformers/logs/


pip3 install omegaconf==2.0.0 pytorch-lightning==1.0.8

cd ..