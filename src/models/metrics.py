import tensorflow as tf
import nltk


def masked_sparse_categorical_accuracy(Y_true, Y_pred):
    """Compute the sparse categorical accuracy, considering the masked timesteps

    Parameters
    ----------
    Y_true : array-like
        An array containing the ground truth for each timestep of each example in the batch,
        where -1 corresponds to padding.
    Y_pred : array-like
        A probability distribution over the vocabulary for each timestep of each example in the
        batch (3-dimensional).

    Returns
    -------
    tf.scalar
        Accuracy in predicting each element of the ground truth sequence
    """
    weight = tf.cast(tf.not_equal(Y_true, -1), tf.float32)
    Y_pred = tf.cast(tf.argmax(Y_pred, axis=-1), tf.float32)
    ones = tf.ones_like(Y_pred)

    accuracy = tf.cast(tf.equal(Y_true, Y_pred), tf.float32)

    return tf.reduce_sum(accuracy * weight) / tf.reduce_sum(ones * weight)


def bleu(captions_true, caption_pred):
    """Evaluation metric for the BLEU score

    Parameters
    ----------
    captions_true : str
        The true caption of the image.
    caption_pred : str
        A caption generated by the model with some sampling technique.
    tokenizer: CustomSpacyTokenizer
        The tokenizer used to split the texts.
    n: int
        N-grams to consider (correspond to computing BLEU-n)
    """
    weights = [
        (1, 0, 0, 0),
        (1./4., 1./4., 1./4., 1./4.)
    ]
    captions_true = [caption.split(" ") for caption in captions_true]
    caption_pred = caption_pred.split(" ")

    return nltk.translate.bleu_score.sentence_bleu(captions_true, caption_pred, weights=weights)
