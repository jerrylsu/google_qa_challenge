import pandas as pd
import numpy as np
import gc
from src.preprocessings.preprocessing import preprocessing
from src.featurizers.featurizer import compute_input_arrays
from src.models.model import create_model
from src.utils.set_gpus import set_gpu

from bert.tokenization import bert_tokenization as tokenization

bert_path = '../data/input/uncased_L-12_H-768_A-12/assets/'
tokenizer = tokenization.FullTokenizer(bert_path + 'vocab.txt')

path = '../data/input/google-quest-challenge/'


def inference():
    print('Starting to predict...')
    df_train = pd.read_csv(path + 'train.csv', header=0, encoding='utf-8')
    df_test = pd.read_csv(path + 'test.csv', header=0, encoding='utf-8')
    input_categories = list(df_train.columns[[1, 2, 5]])
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df_train, df_test, df = preprocessing(df_train, df_test, df, input_categories)
    test_inputs = compute_input_arrays(df_test, ['treated_question_title', 'treated_question_body', 'treated_answer'],
                                       tokenizer, max_sequence_length)
    del df_train
    test_predictions = []
    cols_name_1 = [i for i in df.columns if i.startswith('host_') or i.startswith('cat_')]
    cols_name_2 = ['question_body_num_words', 'answer_num_words', 'question_vs_answer_length', 'q_a_author_same',
                   'answer_user_cat', 'indirect', 'question_count', 'reason_explanation_words', 'choice_words']
    for i in range(1):
        print(f'Loading model-{i}...')
        model_path = f'./models/bert-base-{i}.h5'
        model = create_model()
        model.load_weights(model_path)
        print(f'Model-{i} starts to predict...')
        test_predictions.append(
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.values.html
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_numpy.html
            model.predict(test_inputs
                          + [df_test[cols_name_1].to_numpy()]
                          + [df_test[cols_name_2].to_numpy()],
                          batch_size=8))
        del model
        gc.collect()
    final_predictions = np.mean(test_predictions, axis=0)
    df_sub = pd.read_csv(path + 'sample_submission.csv')
    df_sub.iloc[:, 1:] = final_predictions
    df_sub.to_csv('submission-f.csv', index=False)


def main():
    set_gpu()
    inference()


if '__main__' == __name__:
    main()