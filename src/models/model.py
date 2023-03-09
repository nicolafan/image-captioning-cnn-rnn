import tensorflow as tf
from tensorflow import keras


class ShowAndTell(keras.Model):
    def __init__(
        self,
        n_rnn_neurons,
        img_shape,
        caption_length,
        word_embeddings_size,
        vocab_size,
        stateful=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stateful = stateful

        # encoder
        self.inception1 = keras.applications.InceptionV3(
            include_top=False, weights="imagenet", input_shape=img_shape
        )
        self.inception1.trainable = False
        self.max_pool1 = keras.layers.MaxPool2D((5, 5), name="enc_max_pool1")
        self.flatten1 = keras.layers.Flatten(name="enc_flatten1")
        self.dense1 = keras.layers.Dense(
            n_rnn_neurons, name="enc_dense1"
        )  # since we will init-inject the encoding as RNN state
        self.dropout1 = keras.layers.Dropout(0.3)

        # embedding
        self.embedding1 = keras.layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=word_embeddings_size,
            input_length=caption_length,
            name="embedding1",
        )

        # decoder
        self.lstm1 = keras.layers.LSTM(
            n_rnn_neurons,
            return_sequences=True,
            stateful=self.stateful,
            name="dec_lstm1",
        )
        self.dropout2 = keras.layers.Dropout(0.5)
        self.output1 = keras.layers.TimeDistributed(
            keras.layers.Dense(vocab_size, activation="softmax"), name="output1"
        )

    def call(self, inputs):
        input_img, input_caption = inputs

        # encode
        inception1 = self.inception1(input_img)
        max_pool1 = self.max_pool1(inception1)
        flatten1 = self.flatten1(max_pool1)
        dense1 = self.dense1(flatten1)
        dropout1 = self.dropout1(dense1)

        # embed
        mask = tf.math.not_equal(input_caption, 0)
        embedding1 = self.embedding1(input_caption)

        # decode
        if not self.stateful or self.lstm1.states[0] is None:
            lstm1 = self.lstm1(
                embedding1, mask=mask, initial_state=[dropout1, tf.zeros_like(dropout1)]
            )
        else:
            lstm1 = self.lstm1(embedding1, mask=mask)
        # lstm2 = self.lstm2(lstm1)
        dropout2 = self.dropout2(lstm1)
        output1 = self.output1(dropout2, mask=mask)

        return output1
