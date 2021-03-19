import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from model_lightning import VGG_net

# HIPERPARÁMETROS
max_epochs = 3
batch_size = 16
lr = 1e-3

# Data
print('==> Preparando data..')
transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

dataset_train = CIFAR10(
    root="dataset/train/", 
    train=True,
    transform=transform, 
    download=True
    )
dataset_test = CIFAR10(
    root="dataset/test/", 
    train=False, 
    transform=transform, 
    download=True,
    )

train, val = random_split(dataset_train, [45000, 5000])


print('==> Construyendo red..')
vgg = VGG_net(in_channels=3, num_classes=10, lr=lr, linear_units=100)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_last=True,
    save_top_k=5,
    verbose=False,
)

print('==> Iniciando entrenamiento..')
trainer = pl.Trainer(
    gpus=1, 
    max_epochs=max_epochs,  
    checkpoint_callback=checkpoint_callback,
    progress_bar_refresh_rate=5
    )

trainer.fit(
    vgg, 
    DataLoader(train, batch_size=batch_size), 
    DataLoader(val, batch_size=batch_size)
    )

#tensorboard --logdir lightning_logs/