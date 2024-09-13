conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
conda install xformers -c xformers -y
conda install -c conda-forge ninja -y
pip install fire transformers datasets accelerate trl sentencepiece deepspeed packaging
pip install flash-attn
pip install --no-deps trl peft accelerate bitsandbytes