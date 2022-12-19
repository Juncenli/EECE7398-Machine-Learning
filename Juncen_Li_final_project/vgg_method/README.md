## Using GPUs with PyTorch - Discovery

### Using P100
- `srun --partition=gpu --nodes=1 --pty --gres=gpu:p100:1 --ntasks=1 --mem=4GB --time=04:00:00 /bin/bash`


### Using default environment in Discovery -pytorch_env_training
- `module load anaconda3/2022.01`
- `module load cuda/11.1`
- `source activate pytorch_env_training`

### Train model
- `python Neural_Style.py styleImageName contentImageName`

--- 

python Neural_Style.py Van-Gogh-The-Starry-Night 0

python Neural_Style.py Van-Gogh-The-Starry-Night 1

python Neural_Style.py Van-Gogh-The-Starry-Night 2

python Neural_Style.py Van-Gogh-The-Starry-Night 3

--- 

python Neural_Style.py Mosaic 0

python Neural_Style.py Mosaic 1

python Neural_Style.py Mosaic 2

python Neural_Style.py Mosaic 3

---

python Neural_Style.py NASA-Universe 0

python Neural_Style.py NASA-Universe 1

python Neural_Style.py NASA-Universe 2

python Neural_Style.py NASA-Universe 3

---
python Neural_Style.py Landscape-at-Collioure-Matisse-Les-toits 0

python Neural_Style.py Landscape-at-Collioure-Matisse-Les-toits 1

python Neural_Style.py Landscape-at-Collioure-Matisse-Les-toits 2

python Neural_Style.py Landscape-at-Collioure-Matisse-Les-toits 3
