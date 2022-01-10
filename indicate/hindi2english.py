import signal
from .logging import get_logger
from .base import Base

import tensorflow as tf

from .encoder import Encoder
from .decoder import Decoder, transliterate

logger = get_logger()


class TimeoutException(Exception):  # Custom exception class
    pass


def timeout_handler(signum, frame):  # Custom signal handler
    raise TimeoutException


# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)


class HindiToEnglish(Base):
    MODELFN = "data/model/hindi_to_english/saved_weights/"
    INPUT_VOCAB = "data/model/hindi_to_english/hindi_tokens.json"
    TARGET_VOCAB = "data/model/hindi_to_english/english_tokens.json"

    embedding_dim = 256
    units = 1024
    BATCH_SIZE = 64
    BUFFER_SIZE = 120000

    max_length_input = 47
    max_length_output = 49
    START_TOKEN = "^"
    END_TOKEN = "$"

    weights_loaded = False

    input_lang_tokenizer = None
    target_lang_tokenizer = None
    encoder = None
    decoder = None

    @classmethod
    def transliterate(cls, input, latest=False):
        """
        Transliterate from Hindi to English.
        Args:
            input (str): Hindi text
        Returns:
            output (str): English text
        """

        if not cls.weights_loaded:
            cls.model_path, cls.input_vocab, cls.target_vocab = cls.load_model_data(latest)

            cls.input_lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(cls.input_vocab)
            cls.target_lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(cls.target_vocab)

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
            logger.debug(f"checking weights for encoder {cls.encoder.embedding.variables}")
            logger.debug(f"checking weights for decoder {cls.decoder.embedding.variables}")

            # This is needed to load waits from checkpoint
            example_input_batch = tf.random.uniform(shape=[64, 47])
            sample_hidden = cls.encoder.initialize_hidden_state()
            sample_output, sample_h, sample_c = cls.encoder(example_input_batch, sample_hidden)
            sample_x = tf.random.uniform((cls.BATCH_SIZE, cls.max_length_output))
            cls.decoder.attention_mechanism.setup_memory(sample_output)
            initial_state = cls.decoder.build_initial_state(cls.BATCH_SIZE, [sample_h, sample_c], tf.float32)
            cls.decoder(sample_x, initial_state)

            logger.debug(f"checking weights for encoder after random input {cls.encoder.embedding.variables}")
            logger.debug(f"checking weights for decoder after random input {cls.decoder.embedding.variables}")

            logger.debug(f"Restoring model weights from {cls.model_path}")
            # restore weights from checkpoint
            cls.encoder.load_weights(f"{cls.model_path}/encoder")
            cls.decoder.load_weights(f"{cls.model_path}/decoder")
            logger.debug(f"encoder weights after restoring from checkpoint {cls.encoder.embedding.variables}")
            logger.debug(f"decoder weights after restoring from checkpoint {cls.decoder.embedding.variables}")

            cls.weights_loaded = True

        # Predictions
        # this is needed because sometimes BasicDecoder is going into loop
        # and not exiting for sentences like 'पति स्व.गंगा भगत'
        signal.alarm(10)
        target = ""
        try:
            target = transliterate(
                input,
                cls.units,
                cls.input_lang_tokenizer,
                cls.target_lang_tokenizer,
                cls.encoder,
                cls.decoder,
                cls.max_length_input,
            )
            logger.debug(f"Model predicted {target}")
        except TimeoutException:
            logger.error(f"Giving up on transliterating {input} after 10 seconds")
        except Exception:
            logger.error(f"Not able to transliterate {input}")

        return target
