import tensorflow as tf

def sequence_to_chars(tokenizer, sequence):
    """Convert a sequence of indices back to characters."""
    word_index = tokenizer.word_index
    reverse_map = {val: key for key, val in word_index.items()}
    retext = ''

    # Convert tensor to numpy array if it's a tensor
    if tf.is_tensor(sequence):
        sequence = sequence.numpy()

    for q in sequence:
        q_int = int(q)  # Convert to integer
        if q_int != 0:  # Skip padding token
            retext += reverse_map[q_int]
    return retext

def evaluate_sentence(sentence, units, input_lang_tokenizer, target_lang_tokenizer,
                     encoder, decoder, max_length_input):
    """Evaluate/translate a single sentence."""
    # Convert input sentence to token indices
    inputs = [input_lang_tokenizer.word_index[i] for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_input,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    # Use the same batch size as decoder.batch_sz for consistency
    inference_batch_size = inputs.shape[0]

    inputs = tf.tile(inputs, [inference_batch_size, 1])  # Replicate input to match batch size

    # Initialize encoder state
    enc_start_state = [tf.zeros((inference_batch_size, units)),
                      tf.zeros((inference_batch_size, units))]

    # Get encoder output
    enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)

    # Initialize decoder with encoder's final state
    dec_state = [enc_h, enc_c]

    # Prepare decoder input with start token
    dec_input = tf.fill([inference_batch_size, 1],
                       target_lang_tokenizer.word_index['^'])

    outputs = []

    # Decoding loop
    for t in range(decoder.max_length_output):
        # Call decoder
        predictions, dec_state, _ = decoder(dec_input, dec_state, enc_out)

        # Get predicted token (use first batch item since all are the same)
        predicted_id = tf.argmax(predictions, axis=-1).numpy()[0, -1]

        # Append prediction
        outputs.append(predicted_id)

        # Break if end token is predicted
        if predicted_id == target_lang_tokenizer.word_index['$']:
            break

        # Update decoder input
        dec_input = tf.fill([inference_batch_size, 1], predicted_id)

    return tf.convert_to_tensor(outputs)


def translate(sentence, units, input_lang_tokenizer, target_lang_tokenizer,
             encoder, decoder, max_length_input):
    """Translate a sentence from source to target language."""
    result = evaluate_sentence(sentence,
                             units,
                             input_lang_tokenizer,
                             target_lang_tokenizer,
                             encoder,
                             decoder,
                             max_length_input)

    # Convert the output tokens back to characters
    translated_text = sequence_to_chars(target_lang_tokenizer, result)

    # Remove the end token and return
    return translated_text.strip('$')
