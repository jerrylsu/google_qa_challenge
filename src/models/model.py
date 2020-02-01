import tensorflow as tf
import tensorflow_hub as hub

max_sequence_length = 512
bert_path = '../data/input/uncased_L-12_H-768_A-12/assets/'


# def create_model():
#     input_word_ids = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32, name='input_word_ids')
#     input_masks = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32, name='input_masks')
#     input_segments = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32, name='input_segments')
#     bert_layer = hub.KerasLayer(bert_path, trainable=True)
#     _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])
#     x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_sequence_length))(sequence_output)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     out = tf.keras.layers.Dense(30, activation='sigmoid', name='dense_output')(x)
#
#     model = tf.keras.models.Model([input_word_ids, input_masks, input_segments], out)
#     print(model.summary())
#     return model


def create_model():
    input_word_ids = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32, name='input_segments')

    # input_categories = tf.keras.layers.Input((len(cols_name),), dtype=tf.float32, name='input_categorical')
    input_new_features = tf.keras.layers.Input((9,), dtype=tf.float32, name='input_new_features')

    bert_layer = hub.KerasLayer(bert_path, trainable=True)
    _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])
    x_mean = tf.keras.layers.GlobalMaxPool1D()(sequence_output)
    x_max = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Concatenate()([x_mean, x_max])
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_sequence_length))(sequence_output)
    x = tf.keras.layers.Concatenate()([x, input_new_features])
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(30, activation='sigmoid', name='dense_output')(x)

    model = tf.keras.models.Model([input_word_ids, input_masks, input_segments, input_new_features],
                                  out)
    print(model.summary())
    return model
