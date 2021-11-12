from .base import Base

from keras.models import Model
from keras.layers import Input

import numpy as np


class HindiToEnglish(Base):
    MODELFN = "data/model/hindi_to_english/model.h5"
    INPUT_VOCAB = "data/model/hindi_to_english/input_token_index.json"
    TARGET_VOCAB = "data/model/hindi_to_english/target_token_index.json"

    LATENT_DIM = 256
    MAX_ENCODER_SEQ_LENGTH = 47
    MAX_DECODER_SEQ_LENGTH = 49
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
            cls.model, cls.input_vocab, cls.target_vocab = cls.load_model_data(latest)

        # Predictions
        encoder_inputs = cls.model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = cls.model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]

        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = cls.model.input[1]  # input_2

        decoder_state_input_h = Input(shape=(cls.LATENT_DIM,))
        decoder_state_input_c = Input(shape=(cls.LATENT_DIM,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_lstm = cls.model.layers[3]  # lstm_2
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

        decoder_states = [state_h_dec, state_c_dec]

        decoder_dense = cls.model.layers[-1]
        decoder_outputs = decoder_dense(decoder_outputs)

        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        reverse_target_char_index = dict((i, char) for char, i in cls.target_vocab.items())

        num_encoder_tokens = len(cls.input_vocab)
        num_decoder_tokens = len(cls.target_vocab)
        # convert for encoding

        input_data = np.zeros((1, cls.MAX_ENCODER_SEQ_LENGTH, num_encoder_tokens), dtype="float32")
        for t, char in enumerate(input):
            input_data[0, t, cls.input_vocab[char]] = 1.0

        # Encode the input as state vectors.
        states_value = encoder_model(input_data)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens), dtype="float32")
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, cls.target_vocab["^"]] = 1.0

        decoded_sentence = ""

        while True:
            output_tokens, d_h, d_c = decoder_model.predict([target_seq] + states_value)
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "$" or len(decoded_sentence) > cls.MAX_DECODER_SEQ_LENGTH:
                break

            decoded_sentence += sampled_char

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [d_h, d_c]

        return decoded_sentence
