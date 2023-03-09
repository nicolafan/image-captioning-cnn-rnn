import tensorflow as tf


def masked_sparse_categorical_accuracy(Y_true, Y_pred):
    weight = tf.cast(tf.not_equal(Y_true, -1), tf.float32)
    Y_pred = tf.cast(tf.argmax(Y_pred, axis=-1), tf.float32)
    ones = tf.ones_like(Y_pred)

    accuracy = tf.cast(tf.equal(Y_true, Y_pred), tf.float32)

    return tf.reduce_sum(accuracy * weight) / tf.reduce_sum(ones * weight)
