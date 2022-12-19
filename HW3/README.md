## Using GPUs with PyTorch - Discovery

### Using P100
- `srun --partition=gpu --nodes=1 --pty --gres=gpu:p100:1 --ntasks=1 --mem=4GB --time=04:00:00 /bin/bash`


### Load Modules
- `module load cuda/11.3`

- `module load anaconda3/2022.01`

- `module load python/3.7.0`
  
### Create an environment
- `conda create --name pytorch_env python=3.7 anaconda -y`


### Activate environment
- `source activate pytorch_env`

### Set up
- `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y`
-  `pip install nltk`

### Check pytorch
- `python -c'import torch; print(torch.cuda.is_available())'`

### Or using default environment in Discovery -pytorch_env_training
- `module load anaconda3/2022.01`
- `module load cuda/11.1`
- `source activate pytorch_env_training`

### Train model
- `python NMT.py train`

### Test Model
- `python NMT.py test`

### Translate
- `python NMT.py translate`