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
        model = keras.models.model_from_json(
            model_config_file.read(), custom_objects={"ShowAndTell": ShowAndTell}
        )
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
    plt.imshow((image / 255.0).numpy())
    plt.show()
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(
        image
    )  # change it if CNN changes
    return image


def predict(model, image, tokenizer):
    next_token_idx = tokenizer.vocab["<start>"]
    result = [next_token_idx]

    while next_token_idx != tokenizer.vocab["<end>"] and len(result) <= 16:
        seq = [next_token_idx] + [0] * 15
        image_inp = tf.expand_dims(image, 0)
        seq_inp = tf.constant([seq], dtype=tf.int32)

        distribution = model((image_inp, seq_inp), training=False)[0][
            0
        ]  # first example, first word
        next_token_idx = tf.argmax(distribution).numpy() + 1
        result.append(next_token_idx)

    model.reset_states()
    return tokenizer.sequence_to_text(result)


def predict(model: ShowAndTell, image, tokenizer, beam_width=3):
    next_token_idx = tokenizer.vocab["<start>"]
    initial_hypothesis = {"seq": [next_token_idx], "score": 0.0, "norm_score": 0.0}
    beam = [initial_hypothesis]
    image_inp = tf.expand_dims(image, 0)

    for l in range(1, model.caption_length):
        candidates = []
        for hypo in beam:
            if hypo["seq"][-1] == tokenizer.vocab["<end>"]:
                continue
            seq = hypo["seq"] + [0] * (model.caption_length - len(hypo["seq"]))
            seq_inp = tf.constant([seq], dtype=tf.int32)

            distribution = model((image_inp, seq_inp), training=False)[0][
                l - 1
            ].numpy()  # frist batch, first word
            top_indices = np.argsort(distribution)[-beam_width:]
            top_words = [
                int(i) + 1 for i in top_indices
            ]  # +1 because model outputs are decreased by one
            top_probs = distribution[top_indices]

            # add the candidates to the list
            for word, prob in zip(top_words, top_probs):
                candidate_seq = hypo["seq"] + [word]
                candidate_score = hypo["score"] + np.log(prob)
                candidate_norm_score = candidate_score / l

                candidate = {
                    "seq": candidate_seq,
                    "score": candidate_score,
                    "norm_score": candidate_norm_score,
                }
                candidates.append(candidate)

        # keep the top beam_width candidates based on their score
        beam = [hypo for hypo in beam if hypo["seq"][-1] == tokenizer.vocab["<end>"]]
        beam = sorted(beam + candidates, key=lambda x: x["norm_score"], reverse=True)[
            :beam_width
        ]

    return tokenizer.sequence_to_text(beam[0]["seq"])


@click.command()
@click.option(
    "--model_filename",
    default="",
    help="Filename (without extension) of the model config and weights to load",
)
def main(model_filename):
    data_dir = PROJECT_DIR / "data"
    tokenizer = CustomSpacyTokenizer.from_json()

    # load the iamges from /data/custom
    image_filenames = [
        os.path.join(data_dir / "custom", f)
        for f in os.listdir(data_dir / "custom")
        if f.endswith(".jpg")
    ]
    images = [load_image(filename) for filename in image_filenames]

    model = build_prediction_model(model_filename)
    for image in images:
        result = predict(model, image, tokenizer, beam_width=10)
        print(result)


if __name__ == "__main__":
    main()
