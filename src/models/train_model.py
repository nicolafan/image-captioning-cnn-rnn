import os
import json
import time
import click
from pathlib import Path


import logging
from dotenv import find_dotenv, load_dotenv

import tensorflow as tf
from tensorflow import keras

import src.models.metrics as metrics
from src.features.nlp.tokenizer import CustomSpacyTokenizer
from src.models.model import ShowAndTell
from src.models.read_data import read_split_dataset



@click.command()
@click.option(
    "--img", default=299, help="Size of the input images (equal height and width)"
)
@click.option(
    "--n_rnn_neurons",
    default=512,
    help="Size of the RNN states (and of the img encoding)",
)
@click.option("--embedding_size", default=512, help="Size of the word embeddings")
@click.option("--batch_size", default=15, help="Batch size (choose a multiple of 5)")
@click.option("--epochs", default=10, help="Number of epochs")
@click.option(
    "--learning_rate", default=0.0001, help="Learning rate for the Adam optimizer"
)
@click.option(
    "--model_filename",default=None, help="File name of the saved model"
)
def main(img, n_rnn_neurons, embedding_size, batch_size, epochs, learning_rate,model_filename):
    tf.random.set_seed(42)
    logger = logging.getLogger(__name__)
    
    
    
    project_dir = Path(__file__).resolve().parents[2]
    models_dir = project_dir / "models"

    # train config
    img_shape = (img, img, 3)
    tokenizer = CustomSpacyTokenizer.from_json()
    caption_length = tokenizer.max_len
    vocab_size = tokenizer.vocab_size
    
    # directory path for model and configs
    weights_dir = models_dir / "weights"
    os.makedirs(weights_dir, exist_ok=True) 
    config_dir = models_dir / "config"
    os.makedirs(config_dir, exist_ok=True)
    
    

    # read tf.data.Datasets
    train_dataset = read_split_dataset("train", img_shape, caption_length, batch_size)
    val_dataset = read_split_dataset("val", img_shape, caption_length, batch_size)
    
    if model_filename:
        # load config from saved json file and load model from saved h5 file
        try:
            with open(f"{os.path.abspath(config_dir)}{os.sep}{model_filename}.json",'r') as json_file:
                json_savedModel = json.load(json_file)
            logger.info(f"Checkpoint loaded from :{os.path.abspath(config_dir)}{os.sep}{model_filename}.json")

            model = ShowAndTell(**json_savedModel["config"])
            model.build(input_shape=[(None,) + img_shape, (None, caption_length)])
            model.load_weights(f"{os.path.abspath(weights_dir)}{os.sep}{model_filename}.h5")
            logger.info(f"Model loaded from :{os.path.abspath(weights_dir)}{os.sep}{model_filename}.h5")
        except Exception as e:
            logger.info(e)
            raise
        
    else:
        # create model - show and tell paper configuration
        model = ShowAndTell(
            n_rnn_neurons, img_shape, caption_length, embedding_size, vocab_size
        )
        model.build(input_shape=[(None,) + img_shape, (None, caption_length)])
        
        
    model.summary()
  
    # train model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=-1),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[metrics.masked_sparse_categorical_accuracy],
    )
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    
    # save weights and config
    timestr = time.strftime("%Y%m%d-%H%M%S")

    model_config_filename = f"{os.path.abspath(config_dir)}{os.sep}train_{timestr}.json"
    weights_filename = f"{os.path.abspath(weights_dir)}{os.sep}train_{timestr}.h5"
    model_json_str = model.to_json(indent=4)

    with open(model_config_filename, "w") as file:
        file.write(model_json_str)
    model.save_weights(weights_filename)
   
    logger.info(
        f"saved model config and weights in {model_config_filename} and {weights_filename}"
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
