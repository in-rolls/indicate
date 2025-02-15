import tensorflow as tf

from .logging import get_logger

logger = get_logger()

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz,
                 max_length_input, max_length_output, attention_type='luong'):
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
        self.decoder_rnn = tf.keras.layers.RNN(self.decoder_rnn_cell, return_sequences=True, return_state=True)

        # Create attention mechanism
        # For Luong attention: project the query to match the encoder output's dimension (dec_units)
        if self.attention_type == 'luong':
            self.query_layer = tf.keras.layers.Dense(dec_units)

        self.attention_mechanism = self.build_attention_mechanism(None)

    def build_attention_mechanism(self, memory):
        if self.attention_type == 'bahdanau':
            return tf.keras.layers.AdditiveAttention()
        else:
            return tf.keras.layers.Attention()

    def call(self, inputs, initial_state, encoder_outputs):
        # Get the embeddings of the inputs
        x = self.embedding(inputs)

        if self.attention_type == 'luong':
          query = self.query_layer(x)
        else:
          query = x

        # Feature dimensions to the attention layer
        attention_output, attention_weights = self.attention_mechanism([query, encoder_outputs], return_attention_scores=True)
        # Concatenate the attention output with the LSTM Cell output
        lstm_input = tf.concat([attention_output, x], axis=-1)
        # Process through the LSTM cell
        outputs, state_h, state_c = self.decoder_rnn(lstm_input, initial_state)
        # Pass through the final dense layer
        outputs = self.fc(outputs)
        return outputs, (state_h, state_c), attention_weights

    def build_initial_state(self, batch_sz, encoder_state):
        hidden_state, cell_state = encoder_state
        return [hidden_state, cell_state]
    