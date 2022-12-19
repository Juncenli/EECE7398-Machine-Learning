## Using GPUs with PyTorch - Discovery

- `srun --partition=gpu --nodes=1 --pty --gres=gpu:p100:1 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash`

- `module load cuda/11.1`

- `module load anaconda3/2022.01`
  
- `source activate pytorch_env_training`

- `python -c'import torch; print(torch.cuda.is_available())'`

- `python CNNclassify.py train`

- `python CNNclassify.py test /home/li.junce/EECE7398/HW2/testingImages/truck.png`

The reason for torch.cuda.is_available() resulting False is the incompatibility between the versions of pytorch and cudatoolkit.

