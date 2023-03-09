import os
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

import models.metrics as metrics
from features.build_features import CustomSpacyTokenizer
from models.model import ShowAndTell
from models.read_data import read_split_dataset


def main():
    tf.random.set_seed(42)
    project_dir = Path(__file__).resolve().parents[2]
    processed_data_dir = project_dir / "data" / "processed"
    models_dir = project_dir / "models"

    # train config
    img_shape = (299, 299, 3)
    batch_size = 15  # a multiple of five is preferable
    epochs = 10
    tokenizer_json_path = processed_data_dir / "tokenizer.json"
    tokenizer = CustomSpacyTokenizer.load_from_json(tokenizer_json_path)
    caption_length = tokenizer.max_len

    # read tf.data.Datasets
    train_dataset = read_split_dataset("train", img_shape, caption_length, batch_size)
    val_dataset = read_split_dataset("val", img_shape, caption_length, batch_size)
    test_dataset = read_split_dataset("test", img_shape, caption_length, batch_size)

    # create model - show and tell paper configuration
    model = ShowAndTell(
        512, img_shape, caption_length, 512, tokenizer.vocab_size, stateful=False
    )
    model.build(input_shape=[(None,) + img_shape, (None, caption_length)])
    model.summary()

    # train model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=-1),
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=[metrics.masked_sparse_categorical_accuracy],
    )
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    # save weights
    timestr = time.strftime("%Y%m%d-%H%M%S")
    weights_dir = models_dir / "weights"
    os.makedirs(weights_dir, exist_ok=True)
    weights_filename = f"{os.path.abspath(weights_dir)}/train_{timestr}.h5"
    model.save_weights(weights_filename)
    print(f"saved model weights in {weights_filename}")


if __name__ == "__main__":
    main()
