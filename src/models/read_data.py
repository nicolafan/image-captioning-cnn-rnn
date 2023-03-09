from pathlib import Path
import json
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parents[2]


def read_split_dataset(split, img_shape, caption_length, batch_size):
    processed_data_dir = PROJECT_DIR / "data" / "processed"

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
        img_spec = tf.TensorSpec(shape=img_shape, dtype=tf.float32)
        caption_seq_spec = tf.TensorSpec(shape=(caption_length,), dtype=tf.int32)
        example_spec = ((img_spec, caption_seq_spec), caption_seq_spec)

        image = tf.image.decode_jpeg(example["image_raw"])
        image = tf.image.resize(image, size=img_shape[:-1])
        if image.shape[-1] == 1:  # Grayscale image
            image = tf.image.grayscale_to_rgb(image)
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
    
    parsed_dataset = dataset.map(_parse_example_fn)
    image_captions_dataset = parsed_dataset.map(_to_image_captions_pairs)
    image_caption_dataset = image_captions_dataset.flat_map(
        lambda image, captions: tf.data.Dataset.from_tensor_slices(captions).map(
            lambda caption: _to_input_output_pairs(image, caption)
        )
    )
    return image_caption_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
