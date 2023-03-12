class TokenizerConfig:
    """A configuration for the untrained tokenizer

    This configuration works as a single point for difining
    the tokenizer config, and it will be loaded when instantiating
    a tokenizer using `from_config`.
    """

    MAX_LEN = -1
    VOCAB_SIZE = 8000
    OOV = False
    PAD_SEQUENCES = True
