import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter

from utils import save_checkpoint, load_checkpoint
from model import VGG_net


def train():
    # HIPERPARÁMETROS
    num_epochs = 3
    batch_size = 16
    lr = 1e-3

    # Data
    print('==> Preparando data..')
    transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])

    dataset_train = CIFAR10(
        root="dataset/train/", 
        train=True,
        transform=transform, 
        download=True
        )
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True


    # para los gráficos en tensorboard
    writer = SummaryWriter(f"runs/vgg/")
    step = 0

    # inicializamos el modelo, loss, etc
    model = VGG_net(in_channels=3, num_classes=10, lr=1e-3, linear_units=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if load_model:
        step = load_checkpoint(torch.load("checkpoints/checkpoint.pth"), model, optimizer)

    model.train()

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for idx, (imgs, target) in enumerate(train_loader):
            imgs = imgs.to(device)
            target = target.to(device)

            outputs =  model(imgs)
            # output reshape: (seq_len, N, vocabulary_size)  target shape: (seq_len, N)
            loss =  criterion(outputs, target) 

            writer.add_scalar("Training Loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if idx % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format( epoch + 1, num_epochs, idx, total_step, loss.item()))

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(), 
                "optimizer": optimizer.state_dict(),
                "step": step
            }
            save_checkpoint(checkpoint) 


if __name__ == "__main__":
    train()