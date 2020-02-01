import tensorflow as tf
from transformers import TFBertModel, BertConfig

bert_path = '../data/pretraining_models/bert/bert-base-uncased-tf2-model/'
max_sequence_length = 512


def create_bert_model():
    # Indices of input sequence tokens in the vocabulary.
    input_question_ids = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)
    input_answer_ids = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)

    # Mask to avoid performing attention on padding token indices.
    input_question_masks = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)
    input_answer_masks = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)

    # Segment token indices to indicate first and second portions of the inputs.
    input_question_segments = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)
    input_answer_segments = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)

    config = BertConfig()  # print(config) to see settings
    config.output_hidden_states = False  # Set to True to obtain hidden states
    bert_model = TFBertModel.from_pretrained(bert_path + 'tf_model.h5', config=config)
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    # Sequence of hidden-states at the output of the last layer of the model.
    question_last_hidden_state = bert_model(input_question_ids,
                                            attention_mask=input_question_masks,
                                            token_type_ids=input_question_segments)[0]
    answer_last_hidden_state = bert_model(input_answer_ids,
                                          attention_mask=input_answer_masks,
                                          token_type_ids=input_answer_segments)[0]

    question_mean_pooling = tf.keras.layers.GlobalAveragePooling1D()(question_last_hidden_state)
    question_max_pooling = tf.keras.layers.GlobalMaxPooling1D()(question_last_hidden_state)
    question_pooling = tf.keras.layers.Add()([question_mean_pooling, question_max_pooling])

    answer_mean_pooling = tf.keras.layers.GlobalAveragePooling1D()(answer_last_hidden_state)
    answer_max_pooling = tf.keras.layers.GlobalMaxPooling1D()(answer_last_hidden_state)
    answer_pooling = tf.keras.layers.Add()([answer_mean_pooling, answer_max_pooling])

    question_answer_concat = tf.keras.layers.Concatenate()([question_pooling, answer_pooling])
    x = tf.keras.layers.Dropout(0.2)(question_answer_concat)

    outputs = tf.keras.layers.Dense(30, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[input_question_ids, input_question_masks, input_question_segments,
                                          input_answer_ids, input_answer_masks, input_answer_segments],
                                  outputs=outputs)
    print(model.summary())
    return model

