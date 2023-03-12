import spacy
import json
import os
from pathlib import Path
from features.nlp.config import TokenizerConfig


nlp = spacy.load("en_core_web_sm")


class CustomSpacyTokenizer:
    """Spacy tokenizer to transform text into sequences

    Attributes
    ----------
    max_len : int, optional
        Maximum length of a sequence (the sequence
        will be truncated at the end). If `None` there is no
        truncation. If `-1` the length of the longest text
        in the texts used for fitting will be used. `<start>` and `<end>`
        are included. By default `None`.
    vocab_size : int, optional
        Size of the vocabulary. If `None`, all the unique tokens
        in the fitting texts will be added to the vocabulary. `<start>` and `<end>`
        are included. By default `None`.
    oov : bool
        If `True` it uses the `<oov>` token. Otherwise it skips the out-of-vocabulary token.
    pad_sequences : bool, optional
        If `True` uses zero-padding for the sequences (if `max_len` is specified), by default `True`.
    """

    def __init__(self, max_len=None, vocab_size=None, oov=False, pad_sequences=True):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.oov = oov
        self.pad_sequences = pad_sequences

        self.vocab = {"<start>": 1, "<end>": 2}

        if self.oov:
            self.vocab["<oov>"] = 3

    def fit(self, texts):
        """Fit the tokenizer on a collection of texts

        Parameters
        ----------
        texts : iterable[str]
            A collection of texts as strings.
        """
        counts = {}
        max_text_len = 2

        for text in texts:
            doc = nlp(text)
            text_len = 2
            for token in doc:
                if not token.is_punct:
                    if not token.lower_ in counts:
                        # add lowercase token to the counts
                        counts[token.lower_] = 0
                        text_len += 1
                    counts[token.lower_] += 1
            if text_len > max_text_len:
                max_text_len = text_len

        # set the max_len if it is -1
        if self.max_len == -1:
            self.max_len = max_text_len

        # sort the words by decreasing number of occurrences
        words = [k for (k, _) in sorted(counts.items(), key=lambda x: x[1])]
        words.reverse()

        if self.vocab_size is not None:
            words = words[: self.vocab_size - 2]

        # the most frequent word will have index 3 or 4 (if oov)
        idx = 4 if self.oov else 3
        for word in words:
            self.vocab[word] = idx
            idx += 1

        if self.vocab_size is None:
            self.vocab_size = len(self.vocab)

    def text_to_sequence(self, text):
        """Convert text to sequence of integer words indexes

        Parameters
        ----------
        text : str
            A text.

        Returns
        -------
        list[int]
            A sequence of integers representing the text.
        """
        doc = nlp(text)

        # make a sequence with
        sequence = [self.vocab["<start>"]]
        for token in doc:
            if token.lower_ in self.vocab:
                sequence.append(self.vocab[token.lower_])
            elif self.oov:
                sequence.append(self.vocab["<oov>"])
        sequence.append(self.vocab["<end>"])

        if self.max_len is not None or len(sequence) > self.max_len:
            sequence = sequence[: self.max_len]
            sequence[-1] = self.vocab["<end>"]

        if self.max_len is not None and self.pad_sequences:
            sequence += [0] * (self.max_len - len(sequence))

        return sequence

    def sequence_to_text(self, sequence):
        inv_vocab = dict((v, k) for k, v in self.vocab.items())
        return " ".join(inv_vocab[x] for x in sequence)

    def save_to_json(self):
        """Save the current tokenizer (with vocab) as JSON

        Useful to store the tokenizer after fitting it. This JSON will be
        stored in the same package and will be used to load the tokenizer
        using the `from_json` method.
        """
        current_dir = Path(__file__).resolve().parent
        d = {
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
            "oov": self.oov,
            "pad_sequences": self.pad_sequences,
            "vocab": self.vocab,
        }
        with (current_dir / "tokenizer.json").open("w") as file:
            json.dump(d, file, indent=4)

    @staticmethod
    def from_json():
        """Load a tokenizer from a JSON saved with `save_to_json`

        Returns
        -------
        CustomSpacyTokenizer
            The loaded tokenizer with initialized vocab
        """
        current_dir = Path(__file__).resolve().parent
        with (current_dir / "tokenizer.json").open("r") as file:
            config = json.load(file)
            vocab = config["vocab"]
            del config["vocab"]

            tokenizer = CustomSpacyTokenizer(**config)
            tokenizer.vocab = vocab

            return tokenizer

    @staticmethod
    def from_config():
        """Utility method to load a tokenizer with the params set in `config.py`

        This tokenizer is not trained! If you already fitted a tokenizer, and saved it to
        a JSON file using `save_to_json`, don't call this method.
        This is a method created just to have the tokenizer config in a single point for
        the entire application.
        """
        config = TokenizerConfig()
        return CustomSpacyTokenizer(
            max_len=config.MAX_LEN,
            vocab_size=config.VOCAB_SIZE,
            oov=config.OOV,
            pad_sequences=config.PAD_SEQUENCES,
        )
