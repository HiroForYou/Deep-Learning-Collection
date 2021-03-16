import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from decimal import Decimal
from model import create_vit_classifier
from dataset import dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def predictLabel(model, img_path, checkpoint_dir):

    labels = [
        'avión',
        'automóvil',
        'pájaro',
        'gato',
        'ciervo',
        'perro',
        'rana',
        'caballo',
        'barco',
        'camión'
    ]

    model.load_weights(checkpoint_dir)

    image = Image.open(img_path).convert('RGB')
    plt.imshow(image)
    plt.axis("off")

    image = image.resize((32, 32), Image.ANTIALIAS)
    image = np.asarray(image, dtype=np.float32)
    input_tensor = tf.convert_to_tensor([image], np.float32)

    output_tensor = model.predict(input_tensor)
    output_tensor = tf.nn.softmax(output_tensor)
    arg_pred = tf.argmax(output_tensor, 1)
    
    text =  f"La predicción de ViT es: {labels[int(arg_pred)]}, con {output_tensor.numpy()[0][int(arg_pred)]:.3f} de probabilidad"

    plt.suptitle(text, fontsize=12)
    plt.show()
    

if __name__ == '__main__':

    checkpoint_dir = "./checkpoints/checkpoints"
    img_path = "./src/chiribaya.jpg"

    (_, _, _, _), data_augmentation = dataset()

    vit_classifier = create_vit_classifier(data_augmentation)
    vit_classifier.load_weights(checkpoint_dir)
    predictLabel(vit_classifier, img_path, checkpoint_dir)

    