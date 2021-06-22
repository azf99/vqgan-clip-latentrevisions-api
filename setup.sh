# NOTE: Still for testing on Colab
# TODO: Add all python dependencies in requirements.txt

pip install --no-deps ftfy regex tqdm
pip install kornia
git clone https://github.com/openai/CLIP.git

pip uninstall torchtext --yes

#cd /content/
git clone https://github.com/CompVis/taming-transformers  
cd taming-transformers


# download a VQGAN with a larger codebook (16384 entries)
mkdir -p logs/vqgan_imagenet_f16_16384/checkpoints
mkdir -p logs/vqgan_imagenet_f16_16384/configs

if [ -z "$(ls -A logs/vqgan_imagenet_f16_16384/checkpoints/)" ]; then
  wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt' 
  wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'logs/vqgan_imagenet_f16_16384/configs/model.yaml' 
fi


# !cp /content/drive/MyDrive/vqgan_imagenet_f16_16384-20210325T002625Z-001.zip /content/vq.zip
# !unzip /content/vq.zip -d /content/taming-transformers/logs/


pip install omegaconf==2.0.0 pytorch-lightning==1.0.8

cd ..