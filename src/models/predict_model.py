import tensorflow as tf
from models.model import ShowAndTell
from features.build_features import CustomSpacyTokenizer
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[2]

def build_prediction_model():
    model = ShowAndTell(
        n_rnn_neurons=512,
        img_shape=(299, 299, 3),
        caption_length=16,
        word_embeddings_size=512,
        vocab_size=8000,
        stateful=True
    )
    weights_dir = PROJECT_DIR / "models" / "weights"
    last_weights_file = os.listdir(weights_dir)[-1]
    model.build(input_shape=[(1,) + (299, 299, 3), (1, 16)])
    model.load_weights(f"{os.path.abspath(weights_dir / last_weights_file)}", by_name=True)
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

        distribution = model((image_inp, seq_inp))[0][0] # first example, first word
        next_token_idx = tf.argmax(distribution).numpy() + 1
        result.append(next_token_idx)


    model.reset_states()
    return tokenizer.sequence_to_text(result)

        
def main():
    processed_data_dir = PROJECT_DIR / "data" / "processed"
    tokenizer_json_path = processed_data_dir / "tokenizer.json"
    tokenizer = CustomSpacyTokenizer.load_from_json(tokenizer_json_path)

    filenames = [os.path.join(processed_data_dir / "custom", f) for f in os.listdir(processed_data_dir / "custom") if f.endswith('.jpg')]
    images = [load_image(filename) for filename in filenames]

    model = build_prediction_model()
    for image in images:
        result = predict(model, image, tokenizer)
        print(result)


if __name__ == "__main__":
    main()