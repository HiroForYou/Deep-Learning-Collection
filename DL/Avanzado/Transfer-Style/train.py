import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image

from model import VGG_styler

# CONFIGURACIÓN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 500

# HIPERPARÁMETROS
max_epochs = 3
lr = 1e-3
total_steps = 6000
alpha = 1
beta = 1e-2

# Data
print('==> Preparando data..')
transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def load_image(image_name):
    image = Image.open(image_name)
    image = transform(image).unsqueeze(0)
    return image.to(device)

original_img = load_image("src/wiki.jpg")
style_img = load_image("src/style.jpg")


# Imagen inicial generada como ruido blanco o clon de la imagen original.
# Clon parece funcionar bien

#generated = torch.randn(original_img.shape, device=device, requires_grad=True)
generated_init = original_img.clone().requires_grad_(True)
optimizer = optim.Adam([generated_init], lr=lr)

print('==> Construyendo red..')
model = VGG_styler().to(device).eval()

print('==> Iniciando entrenamiento..')
for step in range(total_steps):
    # obtenemos las características convolucionales de las capas 
    # especificadas
    generated_features = model(generated_init)
    original_img_features = model(original_img)
    style_features = model(style_img)

    # Loss es 0 inicialmente
    style_loss = original_loss = 0

    # iteramos sobre todas las características
    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features
    ):

        # batch_size es 1
        _, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)
        # Calculamos la matriz Gram, de la imagen generada
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )
        # Calculamos la matriz Gram, de la imagen estilo
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (step + 1) % 10 == 0:
        print(f"Step {step+1}/{total_steps} Loss: {total_loss.item()}")
    if (step + 1) % 200 == 0:
        save_image(generated_init, "src/wiki_style.png")