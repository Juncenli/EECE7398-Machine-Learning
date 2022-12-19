import torch

PATH = './model/model.pth'
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
BATCH_SIZE=64
EPOCH=10
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"