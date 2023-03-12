import json
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from features.nlp.tokenizer import CustomSpacyTokenizer
from models.model import ShowAndTell

PROJECT_DIR = Path(__file__).resolve().parents[2]


def build_prediction_model(model_config_filename):
    model_config_dir = PROJECT_DIR / "models" / "config"
    weights_dir = PROJECT_DIR / "models" / "weights"
    if model_config_filename == "":
        last_name = os.listdir(model_config_dir)[-1]
        model_config_path = model_config_dir / last_name
        weights_path = weights_dir / f"{last_name.removesuffix('.json')}.h5"
    else:
        model_config_path = model_config_dir / f"{model_config_filename}.json" 
        weights_path = weights_dir / f"{model_config_filename}.h5"

    with model_config_path.open("r") as model_config_file:
        model_config = json.load(model_config_file)["config"]
        model_config["mode"] = "inference"
        model = ShowAndTell.from_config(model_config)
        # build the model with the input shapes
        model.build(input_shape=[[1] + model.img_shape, [1, model.caption_length]])
        # load corresponding weights
        model.load_weights(weights_path, by_name=True)
        model.summary()
        return model


def load_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=(299, 299))
    plt.imshow((image / 255.).numpy())
    plt.show()
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(image) # change it if CNN changes
    return image


def predict(model, image, tokenizer):
    next_token_idx = tokenizer.vocab["<start>"]
    result = [next_token_idx]

    while next_token_idx != tokenizer.vocab["<end>"] and len(result) <= 16:
        seq = [next_token_idx] + [0]*15
        image_inp = tf.expand_dims(image, 0)
        seq_inp = tf.constant([seq],  dtype=tf.int32)

        distribution = model((image_inp, seq_inp), training=False)[0][0] # first example, first word
        next_token_idx = tf.argmax(distribution).numpy() + 1
        result.append(next_token_idx)


    model.reset_states()
    return tokenizer.sequence_to_text(result)

@click.command()
@click.option("--model_filename", default="", help="Filename (without extension) of the model config and weights to load")        
def main(model_filename):
    data_dir = PROJECT_DIR / "data"
    tokenizer = CustomSpacyTokenizer.from_json()

    # load the iamges from /data/custom
    image_filenames = [os.path.join(data_dir / "custom", f) for f in os.listdir(data_dir / "custom") if f.endswith('.jpg')]
    images = [load_image(filename) for filename in image_filenames]

    model = build_prediction_model(model_filename)
    for image in images:
        result = predict(model, image, tokenizer)
        print(result)


if __name__ == "__main__":
    main()
