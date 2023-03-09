# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import tensorflow as tf
from features.build_features import CustomSpacyTokenizer
import pandas as pd
import os
import math


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _tensor_feature(value):
    """Returns a feature tensor"""
    t = tf.constant(value)
    serialized_t = tf.io.serialize_tensor(t)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_t.numpy()]))


def image_example(image_string, captions, tokenizer):
    """Create an image example

    An image example for the TFRecord is made of the image string encoding,
    image information such as height and width and the integer sequences corresponding
    to the captions of the image.

    Parameters
    ----------
    image_string : str
        The image string encoding.
    captions : list[str]
        The image captions.
    tokenizer : CustomSpacyTokenizer
        The trained tokenizer to transform the captions into sequences.

    Returns
    -------
    tf.train.Example
        A training example for the TFRecords.
    """
    image_shape = tf.io.decode_jpeg(image_string).shape
    caption_seqs = [
        tokenizer.text_to_sequence(caption) for caption in captions
    ]  # tokenize the textual caption

    feature = {
        "height": _int64_feature(image_shape[0]),
        "width": _int64_feature(image_shape[1]),
        "depth": _int64_feature(image_shape[2]),
        "caption_seqs": _tensor_feature(caption_seqs),
        "image_raw": _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def save_tf_records(split, input_dir, output_dir, captions_df, tokenizer):
    """Convert images and related captions and save them to TFRecords

    Parameters
    ----------
    split : str
        Identification of the split (train, test, val).
    input_dir : Path
        Directory in which the data is initially stored (should be /data/raw).
    output_dir : Path
        Directory in which to save the records of the split (should be /data/processed).
    captions_df : DataFrame
        A dataframe having two columns: `image`, with the filename of the images and `caption`
        containing at each row a pair image-caption (there could be multiple rows for a single)
        image if it has multiple captions.
    tokenizer : CustomSpacyTokenizer
        The tokenizer to transform the strings (captions) to sequences.
    """
    split_file = (input_dir / f"{split}_split_filenames.txt").open("r")
    split_ids = [line.strip() for line in split_file.readlines()]

    os.makedirs(output_dir / split, exist_ok=True)

    # make multiple TFRecords for the images in the split, where each record contains 200 images
    for file_n in range(math.ceil(len(split_ids) / 200)):
        next_rec_ids = split_ids[file_n * 200 : file_n * 200 + 200]
        record_file = output_dir / split / f"images_{str(file_n).zfill(3)}.tf_records"
        with tf.io.TFRecordWriter(str(record_file)) as writer:
            for image_id in next_rec_ids:
                image_captions = captions_df[captions_df["image"] == image_id][
                    "caption"
                ].tolist()
                if image_id in split_ids:
                    image_string = open(
                        input_dir / "images" / f"{str(image_id)}", "rb"
                    ).read()
                    tf_example = image_example(image_string, image_captions, tokenizer)
                    writer.write(tf_example.SerializeToString())


@click.command()
def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final dataset from raw data")

    # setup the dirs
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = project_dir / "data" / "raw"
    processed_data_dir = project_dir / "data" / "processed"

    # create and fit the tokenizer
    tokenizer = CustomSpacyTokenizer(
        max_len=-1, vocab_size=8000, oov=False, pad_sequences=True
    )
    captions_df = pd.read_csv(raw_data_dir / "captions.txt", sep=",")

    logger.info("fitting tokenizer on captions")
    tokenizer.fit(captions_df["caption"].tolist())
    tokenizer.save_to_json(processed_data_dir)
    logger.info("tokenizer trained!")

    # create the tf records
    for split in "train", "val", "test":
        logger.info(f"making {split} records")
        save_tf_records(split, raw_data_dir, processed_data_dir, captions_df, tokenizer)
        logger.info(f"{split} records ready!")

    logger.info("processed dataset TFRecords created!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
