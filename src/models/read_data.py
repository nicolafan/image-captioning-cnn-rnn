import json
from pathlib import Path

import tensorflow as tf


def read_split_dataset(split, img_shape, caption_length, batch_size):
    """Prepare a TFDataset out of the TFRecords corresponding to `split`

    The dataset will be ready for training a model over it.
    It will batch the data with the specified`batch_size`, and will be composed
    of `(inputs, output)` tuples where:

    * `inputs` consists in a tuple with two elements: the image (a `tf.float32` tensor
    of an RGB image having `img_shape` and already preprocessed to work with Inceptionv3), and
    the caption sequence, a `tf.int32` tensor of `caption_length` (fixed for each element of the dataset)
    where each index corresponds to the index of the word in the tokenizer and which uses 0-padding.
    * `outputs` is a `tf.int32` tensor having `caption_length` and corresponding to the caption sequence
    of integers, slid of one place to the left and where each index is decreased by 1 (this is to avoid
    that the 0-th neuron in the softmax layer would be associated to padding). In this way we have that
    `-1` corresponds to padding tokens that should be ignored during the computation of the loss.


    Parameters
    ----------
    split : str
        `"train"`, `"val"` or `"test"`.
    img_shape : tuple[int]
        The required image shape.
    caption_length : int
        Maximum length of a sequence.
    batch_size : int
        The batch size.

    Returns
    -------
    Tensorflow MapDataset
        A tf.data.Dataset containing `(inputs, output)` pairs, ready to be given to a model.
    """
    project_dir = Path(__file__).resolve().parents[2]
    processed_data_dir = project_dir / "data" / "processed"

    split_dir = processed_data_dir / split
    filenames = tf.data.Dataset.list_files(str(split_dir / "*.tf_records"))
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)

    example_feature_description = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
        "caption_seqs": tf.io.FixedLenFeature([], tf.string),
        "image_raw": tf.io.FixedLenFeature([], tf.string),
    }

    # parse the examples using the description
    def _parse_example_fn(example_proto):
        return tf.io.parse_single_example(example_proto, example_feature_description)

    # make the dataset as a set of ((image, caption_seq), slid_caption_seq) pairs
    # where the slid_caption_seq is the caption_seq slided by one to the left, to be
    # used as the reference caption for the results
    # because of 0-indexing we also have to decrease by 1 the slid_caption_seq
    def _to_image_captions_pairs(example):
        image = tf.image.decode_jpeg(example["image_raw"], channels=3) # last dim of img_shape is 3
        image = tf.image.resize(image, size=img_shape[:-1])
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.inception_v3.preprocess_input(image)

        caption_seqs = tf.io.parse_tensor(example["caption_seqs"], out_type=tf.int32)
        caption_seqs = tf.ensure_shape(caption_seqs, [None, caption_length])

        return image, caption_seqs

    def _to_input_output_pairs(image, caption_seq):
        slid_caption_seq = tf.roll(caption_seq, shift=-1, axis=0)
        slid_caption_seq = tf.tensor_scatter_nd_update(
            slid_caption_seq, [[caption_length - 1]], [0]
        )
        slid_caption_seq -= 1

        return (image, caption_seq), slid_caption_seq
    
    # parse the examples
    parsed_dataset = dataset.map(_parse_example_fn)
    # associate n captions in a single tensor to their image
    image_captions_dataset = parsed_dataset.map(_to_image_captions_pairs)
    # split the captions to make different samples for each image-caption pair
    # and make the ground truth
    image_caption_dataset = image_captions_dataset.flat_map(
        lambda image, captions: tf.data.Dataset.from_tensor_slices(captions).map(
            lambda caption: _to_input_output_pairs(image, caption)
        )
    )
    
    return image_caption_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
