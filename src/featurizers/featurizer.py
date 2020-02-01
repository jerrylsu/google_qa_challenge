import numpy as np
from tqdm import tqdm


def convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    def get_ids(sentence1, sentence2, max_length, truncation_strategy='longest_first'):
        res_dict = tokenizer.encode_plus(text=sentence1,
                                         text_pair=sentence2,
                                         add_special_tokens=True,
                                         max_length=max_length,
                                         truncation_strategy=truncation_strategy)

        input_ids, input_masks, input_segments = res_dict["input_ids"], res_dict['attention_mask'], res_dict["token_type_ids"]

        padding_length, padding_id = max_length - len(input_ids), tokenizer.pad_token_id

        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids_q, input_masks_q, input_segments_q = get_ids(title + ' ' + question, None, max_sequence_length)
    input_ids_a, input_masks_a, input_segments_a = get_ids(answer, None, max_sequence_length)

    return [input_ids_q, input_masks_q, input_segments_q, input_ids_a, input_masks_a, input_segments_a]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_question_ids, input_question_masks, input_question_segments = [], [], []
    input_answer_ids, input_answer_masks, input_answer_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        title, question, answer = instance.question_title, instance.question_body, instance.answer

        q_ids, q_masks, q_segments, a_ids, a_masks, a_segments = convert_to_transformer_inputs(title,
                                                                                               question,
                                                                                               answer,
                                                                                               tokenizer,
                                                                                               max_sequence_length)

        input_question_ids.append(q_ids)
        input_question_masks.append(q_masks)
        input_question_segments.append(q_segments)

        input_answer_ids.append(a_ids)
        input_answer_masks.append(a_masks)
        input_answer_segments.append(a_segments)

    return [np.asarray(input_question_ids, dtype=np.int32),
            np.asarray(input_question_masks, dtype=np.int32),
            np.asarray(input_question_segments, dtype=np.int32),
            np.asarray(input_answer_ids, dtype=np.int32),
            np.asarray(input_answer_masks, dtype=np.int32),
            np.asarray(input_answer_segments, dtype=np.int32)]


def compute_xlnet_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_question_ids, input_question_masks, input_question_segments = [], [], []
    input_answer_ids, input_answer_masks, input_answer_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        title, question, answer = instance.question_title, instance.question_body, instance.answer

        q_ids, q_masks, q_segments, a_ids, a_masks, a_segments = convert_to_transformer_inputs(title,
                                                                                               question,
                                                                                               answer,
                                                                                               tokenizer,
                                                                                               max_sequence_length)

        input_question_ids.append(q_ids)
        input_question_masks.append(q_masks)
        input_question_segments.append(q_segments)

        input_answer_ids.append(a_ids)
        input_answer_masks.append(a_masks)
        input_answer_segments.append(a_segments)

    return [np.asarray(input_question_ids, dtype=np.int32),
            np.asarray(input_question_masks, dtype=np.float32),
            np.asarray(input_question_segments, dtype=np.int32),
            np.asarray(input_answer_ids, dtype=np.int32),
            np.asarray(input_answer_masks, dtype=np.float32),
            np.asarray(input_answer_segments, dtype=np.int32)]


def compute_label_arrays(df, columns):
    return np.asarray(df[columns])
