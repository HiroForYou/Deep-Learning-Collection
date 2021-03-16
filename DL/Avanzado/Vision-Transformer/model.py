import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from dataset import dataset
import matplotlib.pyplot as plt

# HIPERPARÁMETROS PARA EL modelo
num_classes = 10 # CIFAR10
input_original_shape = (32, 32, 3)

image_size = 72  # Cambiaremos el tamaño de las imágenes de entrada a este tamaño
patch_size = 6  # Tamaño de los parches que se extraerán de las imágenes de entrada
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Tamaño de las capas del transformer
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Tamaño de las capas densas del clasificador final

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tfa.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_classifier(data_augmentation):

    inputs = layers.Input(shape=input_original_shape)
    # Aumento de datos
    augmented = data_augmentation(inputs)
    # Creamos los parches
    patches = Patches(patch_size)(augmented)
    # Codificamos los parches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Creamos múltiples capas del bloque Transformer
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Creamos una capa multi-head attention
        '''
        # solo soportado para TF 2.4
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        '''

        mha = tfa.layers.MultiHeadAttention(
            head_size=projection_dim, num_heads=num_heads, dropout=0.1
        )
        attention_output = mha([x1, x1])
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Creamos un tensor de forma [batch_size, projection_dim].
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Agregamos la capa mlp
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Clasificamos las salidas
    logits = layers.Dense(num_classes)(features)
    # Creamos el modelo Keras
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


if __name__ == "__main__":

    (x_train, _, _, _), data_augmentation = dataset(image_size=image_size)
    model = create_vit_classifier(data_augmentation)
    
    print("\n\nComprobando funcionamiento de los parches...")
    plt.figure(figsize=(4, 4))
    image = x_train[np.random.choice(range(x_train.shape[0]))]
    plt.imshow(image.astype("uint8"))
    plt.axis("off")

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    patches = Patches(patch_size)(resized_image)
    print(f"Tamaño de la imagen: {image_size} X {image_size}")
    print(f"Tamaño del parche: {patch_size} X {patch_size}")
    print(f"Parche por imagen: {patches.shape[1]}")
    print(f"Elementos por parche: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")

    plt.show()

    print("Comprobando funcionamiento de ViT_Classifier...")
    input_tensor = tf.random.normal([1, 32, 32, 3])
    output_tensor = model.predict(input_tensor)
    print(input_tensor, end="\n\n")
    print(output_tensor, end="\n")
