from .logging import get_logger
from .base import Base

import tensorflow as tf

from .encoder import Encoder
from .decoder import Decoder, translate

logger = get_logger()


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

    model = None

    @classmethod
    def transliterate(cls, input, latest=False):
        """
        Transliterate from Hindi to English.
        Args:
            input (str): Hindi text
        Returns:
            output (str): English text
        """

        if cls.model is None:
            cls.model_path, cls.input_vocab, cls.target_vocab = cls.load_model_data(latest)

        input_lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(cls.input_vocab)
        target_lang_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(cls.target_vocab)

        vocab_inp_size = len(input_lang_tokenizer.word_index) + 1
        vocab_tar_size = len(target_lang_tokenizer.word_index) + 1

        encoder = Encoder(vocab_inp_size, cls.embedding_dim, cls.units, cls.BATCH_SIZE)
        decoder = Decoder(
            vocab_tar_size,
            cls.embedding_dim,
            cls.units,
            cls.BATCH_SIZE,
            cls.max_length_input,
            cls.max_length_output,
            "luong",
        )
        optimizer = tf.keras.optimizers.Adam()
        logger.debug(f"checking weights for encoder {encoder.embedding.variables}")
        logger.debug(f"checking weights for decoder {decoder.embedding.variables}")

        # This is needed to load waits from checkpoint
        example_input_batch = tf.random.uniform(shape=[64, 47])
        sample_hidden = encoder.initialize_hidden_state()
        sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)
        sample_x = tf.random.uniform((cls.BATCH_SIZE, cls.max_length_output))
        decoder.attention_mechanism.setup_memory(sample_output)
        initial_state = decoder.build_initial_state(cls.BATCH_SIZE, [sample_h, sample_c], tf.float32)
        decoder(sample_x, initial_state)

        logger.debug(f"checking weights for encoder after random input {encoder.embedding.variables}")
        logger.debug(f"checking weights for decoder after random input {decoder.embedding.variables}")

        logger.debug(f"Restoring model weights from {cls.model_path}")
        # restore weights from checkpoint
        encoder.load_weights(f"{cls.model_path}/encoder")
        decoder.load_weights(f"{cls.model_path}/decoder")
        logger.debug(f"encoder weights after restoring from checkpoint {encoder.embedding.variables}")
        logger.debug(f"decoder weights after restoring from checkpoint {decoder.embedding.variables}")

        # Predictions
        target = translate(
            input, cls.units, input_lang_tokenizer, target_lang_tokenizer, encoder, decoder, cls.max_length_input
        )
        logger.debug(f"Model predicted {target}")

        return target
