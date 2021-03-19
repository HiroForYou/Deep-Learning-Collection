import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms


def save_checkpoint(state, filename="checkpoints/checkpoint.pth"):
    print("=> Guardando checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Cargando checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step