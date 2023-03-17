import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.features.nlp.tokenizer import CustomSpacyTokenizer
from src.models.model import ShowAndTell
from src.models.utils import load_image_jpeg, build_saved_model


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
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data"
    tokenizer = CustomSpacyTokenizer.from_json()

    # load the iamges from /data/custom
    image_filenames = [
        os.path.join(data_dir / "custom", f)
        for f in os.listdir(data_dir / "custom")
        if f.endswith(".jpg")
    ]
    images = [load_image_jpeg(filename) for filename in image_filenames]

    model = build_saved_model(model_filename, mode="inference")

    predictions = []
    for image in images:
        result = predict(model, image, tokenizer, beam_width=3)
        predictions.append((image, result))

    # visualize results
    # Create a grid of subplots
    fig, axs = plt.subplots(math.ceil(len(images) / 3), 3, figsize=(5, 5))

    if len(images) == 1:
        image, caption = predictions[0]
        axs.imshow(image)
        axs.set_title(caption)
        axs.axis("off")
    else:
        i = 0
        j = 0
        for image, caption in predictions:
            # Plot each image in a separate subplot with a caption
            axs[i][j].imshow(image)
            axs[i][j].set_title(caption, fontsize="8")
            axs[i][j].axis('off')
            axs[i][j].axis("off")
            j += 1
            if j >= 3:
                j = 0
                i += 1

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
