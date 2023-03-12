import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from models.model import ShowAndTell


def load_image_jpeg(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=(299, 299))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(
        image
    )  # change it if CNN changes
    return image


def build_saved_model(model_config_filename):
    project_dir = Path(__file__).resolve().parents[2]
    model_config_dir = project_dir / "models" / "config"
    weights_dir = project_dir / "models" / "weights"
    if model_config_filename == "":
        last_name = os.listdir(model_config_dir)[-1]
        model_config_path = model_config_dir / last_name
        weights_path = weights_dir / f"{last_name.removesuffix('.json')}.h5"
    else:
        model_config_path = model_config_dir / f"{model_config_filename}.json"
        weights_path = weights_dir / f"{model_config_filename}.h5"

    with model_config_path.open("r") as model_config_file:
        model = keras.models.model_from_json(
            model_config_file.read(), custom_objects={"ShowAndTell": ShowAndTell}
        )
        # build the model with the input shapes
        model.build(input_shape=[[1] + model.img_shape, [1, model.caption_length]])
        # load corresponding weights
        model.load_weights(weights_path, by_name=True)
        model.summary()
        return model