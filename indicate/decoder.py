import tensorflow as tf
import tensorflow_addons as tfa

from .logging import get_logger

logger = get_logger()


class Decoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        dec_units,
        batch_sz,
        max_length_input,
        max_length_output,
        attention_type="luong",
    ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(
            self.dec_units, None, self.batch_sz * [self.max_length_input], self.attention_type
        )

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnn_cell, self.attention_mechanism, attention_layer_size=self.dec_units
        )
        return rnn_cell

    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type="luong"):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to
        # max_length_input (for masking purpose)

        if attention_type == "bahdanau":
            return tfa.seq2seq.BahdanauAttention(
                units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length
            )
        else:
            return tfa.seq2seq.LuongAttention(
                units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length
            )

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(
            x, initial_state=initial_state, sequence_length=self.batch_sz * [self.max_length_output - 1]
        )
        return outputs


# Generating text from sequence (mapping back)
def sequence_to_chars(tokenizer, sequence):
    word_index = tokenizer.word_index
    reverse_map = {val: key for key, val in word_index.items()}
    retext = ""
    for q in sequence:
        if q != 0:
            retext += reverse_map[q]
    return retext


def evaluate_sentence(sentence, units, input_lang_tokenizer, target_lang_tokenizer, encoder, decoder, max_length_input):
    logger.debug(f"Evaluating sentence {sentence}")
    # sentence = '^' + sentence.strip() + '$'

    inputs = [input_lang_tokenizer.word_index[i] for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_input, padding="post")
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    logger.debug(f"Inference batch size {inference_batch_size}")

    enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size, units))]
    enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)

    start_tokens = tf.fill([inference_batch_size], target_lang_tokenizer.word_index["^"])
    end_token = target_lang_tokenizer.word_index["$"]

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()
    logger.debug("Initialized greedy sampler")

    # Instantiate BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc)
    logger.debug("Initialized basic decoder")
    # Setup Memory in decoder stack
    decoder.attention_mechanism.setup_memory(enc_out)
    logger.debug("Attention memory setup done.")

    # set decoder_initial_state
    decoder_initial_state = decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)
    logger.debug("Decoder initial state done.")

    # Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder
    # decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this.
    # You only need to get the weights of embedding layer,
    # which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

    decoder_embedding_matrix = decoder.embedding.variables[0]
    logger.debug("Assigned embedded weights")

    outputs, _, _ = decoder_instance(
        decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state
    )
    logger.debug("Got outputs from decoder instance.")
    return outputs.sample_id.numpy()


def transliterate(sentence, units, input_lang_tokenizer, target_lang_tokenizer, encoder, decoder, max_length_input):
    result = ""
    try:
        result = evaluate_sentence(
            sentence, units, input_lang_tokenizer, target_lang_tokenizer, encoder, decoder, max_length_input
        )
        result = sequence_to_chars(target_lang_tokenizer, result[0])
    except Exception:
        logger.error(f"Not able to transliterate {sentence}")
    return result.strip("$")
