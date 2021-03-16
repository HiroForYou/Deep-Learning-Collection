from tensorflow import keras
from tensorflow.keras import layers

def dataset(image_size=72):

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.Resizing(image_size, image_size),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(factor=0.02),
            layers.experimental.preprocessing.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    # calculamos la media y varianza en los datos de entrenamiento, para la normalización
    data_augmentation.layers[0].adapt(x_train)

    return (x_train, y_train, x_test, y_test), data_augmentation

