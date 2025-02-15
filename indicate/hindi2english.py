import json
from .logging import get_logger
from pkg_resources import resource_filename

import tensorflow as tf

from .encoder import Encoder
from .decoder import Decoder
from .utils import translate

from func_timeout import func_timeout, FunctionTimedOut


logger = get_logger()


class HindiToEnglish():
    MODELFN = "data/model/hindi_to_english/saved_weights/"
    INPUT_VOCAB = "data/model/hindi_to_english/hindi_tokens.json"
    TARGET_VOCAB = "data/model/hindi_to_english/english_tokens.json"

    embedding_dim = 256
    units = 1024
    BATCH_SIZE = 64
    BUFFER_SIZE = 120000

    max_length_input = 47
    max_length_output = 173
    START_TOKEN = "^"
    END_TOKEN = "$"

    weights_loaded = False

    input_lang_tokenizer = None
    target_lang_tokenizer = None
    encoder = None
    decoder = None

    @classmethod
    def get_model_path(cls):
        return resource_filename(__name__, cls.MODELFN)

    @classmethod
    def get_input_vocab(cls):
        return resource_filename(__name__, cls.INPUT_VOCAB)

    @classmethod
    def get_target_vocab(cls):
        return resource_filename(__name__, cls.TARGET_VOCAB)

    @classmethod
    def transliterate(cls, input: str) -> str:
        """
        Transliterate from Hindi to English.

        Args:
            input (str): Hindi text
        Returns:
            output (str): English text
        """

        if not cls.weights_loaded:
            cls.model_path = cls.get_model_path()
            cls.input_vocab = cls.get_input_vocab()
            cls.target_vocab = cls.get_target_vocab()

            with open(cls.get_input_vocab(), "r", encoding="utf-8") as f: 
                cls.input_lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))

            with open(cls.get_target_vocab(), "r", encoding="utf-8") as f: 
                cls.target_lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))

            vocab_inp_size = len(cls.input_lang_tokenizer.word_index) + 1
            vocab_tar_size = len(cls.target_lang_tokenizer.word_index) + 1

            cls.encoder = Encoder(vocab_inp_size, cls.embedding_dim, cls.units, cls.BATCH_SIZE)
            cls.decoder = Decoder(
                vocab_tar_size,
                cls.embedding_dim,
                cls.units,
                cls.BATCH_SIZE,
                cls.max_length_input,
                cls.max_length_output,
                "luong",
            )

            # This is needed to load waits from checkpoint
            example_input_batch = tf.random.uniform(shape=[cls.BATCH_SIZE, cls.max_length_input])
            sample_hidden = cls.encoder.initialize_hidden_state()
            sample_output, sample_h, sample_c = cls.encoder(example_input_batch, sample_hidden)

            sample_x = tf.random.uniform((cls.BATCH_SIZE, cls.max_length_output))
            initial_state = cls.decoder.build_initial_state(cls.BATCH_SIZE, [sample_h, sample_c])
            cls.decoder(sample_x, initial_state, sample_output)

            logger.debug(f"Restoring model weights from {cls.model_path}")
            # restore weights from checkpoint
            cls.encoder.load_weights(f"{cls.model_path}/encoder.weights.h5")
            cls.decoder.load_weights(f"{cls.model_path}/decoder.weights.h5")

            cls.weights_loaded = True

        # Predictions
        # this is needed because sometimes BasicDecoder is going into loop
        # and not exiting for sentences like 'पति स्व.गंगा भगत'
        # if space is there then split by space and pass each sentence to translate
        if " " in input:
            input = input.split(" ")
        else:
            input = [input]

        # translate each sentence
        output = []
        for inp in input:
            target = ""
            try:
                target = func_timeout(
                    10,
                    translate,
                    args=(
                        inp,
                        cls.units,
                        cls.input_lang_tokenizer,
                        cls.target_lang_tokenizer,
                        cls.encoder,
                        cls.decoder,
                        cls.max_length_input,
                    ),
                )

                logger.debug(f"Model predicted {target}")
            except FunctionTimedOut as fex:
                logger.error(f"Not able to transliterate {input} within 10 seconds, exiting with {fex}")
            except Exception as exe:
                logger.error(f"Not able to transliterate {input} due to {exe}")
            output.append(target)

        return " ".join(output)
