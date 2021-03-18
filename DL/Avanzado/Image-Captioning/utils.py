import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()

    img1 = Image.open("test_images/michi.jpg").convert("RGB")
    test_img1 = transform(
        img1
    ).unsqueeze(0)
    plt.imshow(img1)
    plt.axis("off")
    textTarget1 = f"Example 1 CORRECT: A cat eating a banana"
    textOutput1 =  f"Example 1 OUTPUT: " + " ".join(
        model.caption_image(test_img1.to(device), dataset.vocab))
    textResult1 = textTarget1 + "\n" + textOutput1
    plt.suptitle(textResult1, fontsize=8)
    plt.show()


    img2 = Image.open("test_images/wiki.jpg").convert("RGB")
    test_img2 = transform(
        img2
    ).unsqueeze(0)
    plt.imshow(img2)
    plt.axis("off")

    textTarget2 = f"Example 2 CORRECT: A person standing near a lake"
    textOutput2 =  f"Example 2 OUTPUT: " + " ".join(
        model.caption_image(test_img2.to(device), dataset.vocab))
    textResult2 = textTarget2 + "\n" + textOutput2
    plt.suptitle(textResult2, fontsize=8)
    plt.show()

    model.train()


def save_checkpoint(state, filename="checkpoints/checkpoint.pth"):
    print("=> Guardando checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Cargando checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step