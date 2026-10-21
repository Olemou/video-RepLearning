cd /home/olemou/video-RepLearning
source /home/olemou/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu
python -m scripts.images_to_mp4 \
  --input-root /home/olemou/datasets \
  --output-root /home/olemou/datasets/videos \
  --fps 15