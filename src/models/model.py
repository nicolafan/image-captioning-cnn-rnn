import json

import tensorflow as tf
from tensorflow import keras


class ShowAndTell(keras.Model):
    """Implementation of "Show and tell: A neural image caption generator"

    The RNN layer(s) must be stateful at inference time and not stateful during training;
    this is because during training we will use teacher forcing to train the model, while
    at inference time, we must predict a word at a time and refeed it to the network to 
    get the next word. We will use padding to make this operation as flawless and efficient
    as possible, as if the network was making a single prediction.
    """
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
        self.n_rnn_neurons = n_rnn_neurons
        self.img_shape = img_shape
        self.caption_length = caption_length
        self.word_embeddings_size = word_embeddings_size
        self.vocab_size = vocab_size
        self.stateful = stateful

        # encoder
        self.inception1 = keras.applications.InceptionV3(
            include_top=False, weights="imagenet", input_shape=img_shape
        )
        self.inception1.trainable = False
        self.max_pool1 = keras.layers.MaxPool2D((5, 5), name="enc_max_pool1")
        self.flatten1 = keras.layers.Flatten(name="enc_flatten1")
        # since we will init-inject the encoding as RNN state
        # we need to have an encoding that has the same size as the
        # RNN state
        self.dense1 = keras.layers.Dense(
            n_rnn_neurons, activation="tanh", name="enc_dense1"
        )  
        self.dropout1 = keras.layers.Dropout(0.3) # add a dropout over the encoding

        # embedding
        self.embedding1 = keras.layers.Embedding(
            input_dim=vocab_size + 1, # +1 because 0 is padding
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

        # mask padding and embed
        mask = tf.math.not_equal(input_caption, 0)
        embedding1 = self.embedding1(input_caption)

        # decode
        # we check if the hidden state of the RNN is empty, in which case in a stateful
        # RNN (inference time) we must provide it with the image encoding. Otherwise
        # the state will correspond to the previous prediction so we must keep it
        lstm1_hidden_state = self.lstm1.states[0]
        if lstm1_hidden_state is None or tf.reduce_all(tf.equal(lstm1_hidden_state, tf.constant(0.0))):
            is_lstm1_hidden_empty = True
        else:
            is_lstm1_hidden_empty = False

        if not self.stateful or is_lstm1_hidden_empty:
            lstm1 = self.lstm1(
                embedding1, mask=mask, initial_state=[dropout1, tf.zeros_like(dropout1)]
            )
        else:
            lstm1 = self.lstm1(embedding1, mask=mask)
        # lstm2 = self.lstm2(lstm1) if you need more recurrent layers add them here and manage the state accordingly
        dropout2 = self.dropout2(lstm1)
        output1 = self.output1(dropout2, mask=mask)

        return output1
    
    def get_config(self):
        config = super().get_config()
        show_and_tell_config = {
            "n_rnn_neurons": self.n_rnn_neurons,
            "img_shape": self.img_shape,
            "caption_length": self.caption_length,
            "word_embeddings_size": self.word_embeddings_size,
            "vocab_size": self.vocab_size
        }
        config.update(show_and_tell_config)
        return config
    
    @classmethod
    def from_config(cls, config):
        if config["mode"] == "training":
            del config["mode"]
            return cls(**config, stateful=False)
        elif config["mode"] == "inference":
            del config["mode"]
            return cls(**config, stateful=True)
        else:
            raise ValueError(f"`mode` must be 'training' or 'inference', got {config['mode']} instead")
