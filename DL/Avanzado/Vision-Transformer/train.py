import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from dataset import dataset
from model import create_vit_classifier 

# HIPERPARÁMETROS PARA EL ENTRENAMIENTO
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 10

def run_experiment(model, dataset):

    (x_train, y_train, x_test, y_test) = dataset
    
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_dir = "./checkpoints/checkpoints"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_dir,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
        verbose=1,
    )

    model.load_weights(checkpoint_dir)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history

if __name__ == '__main__':
    data, data_augmentation = dataset()
    vit_classifier = create_vit_classifier(data_augmentation)
    history = run_experiment(vit_classifier, data)