# download a VQGAN with a larger codebook (16384 entries)
echo "Downloading and saving VQGAN weights...."
mkdir -p "./logs/vqgan_imagenet_f16_16384/checkpoints"
mkdir -p "./logs/vqgan_imagenet_f16_16384/configs"

if [ -z "$(ls -A ./logs/vqgan_imagenet_f16_16384/checkpoints/)" ]; then
  wget 'https://mead-model-dependencies.s3.amazonaws.com/last.ckpt' -O './logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt' 
  wget 'https://mead-model-dependencies.s3.amazonaws.com/model.yaml' -O './logs/vqgan_imagenet_f16_16384/configs/model.yaml' 
fi
echo "VQGAN weights saved...."

wget https://mead-model-dependencies.s3.amazonaws.com/karras2019stylegan-ffhq-1024x1024.for_g_all.pt

#sudo add-apt-repository ppa:redislabs/redis
sudo apt update
sudo apt-get install -y redis

pip3 install git+https://github.com/bakztfuture/CLIP.git
pip3 install -r requirements.txt
pip3 uninstall torchtext --yes

#cd /content/
git clone https://github.com/bakztfuture/taming-transformers  
cd taming-transformers

# !cp /content/drive/MyDrive/vqgan_imagenet_f16_16384-20210325T002625Z-001.zip /content/vq.zip
# !unzip /content/vq.zip -d /content/taming-transformers/logs/


#pip3 install omegaconf==2.0.0 pytorch-lightning==1.0.8

cd ..