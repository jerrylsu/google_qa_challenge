import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import GroupKFold
from transformers import BertTokenizer
from preprocessings.preprocessing import preprocessing
from featurizers.featurizer import compute_input_arrays, compute_label_arrays
from models.bert_model import create_bert_model
from utils.set_gpus import set_gpu
from utils.custom_callbacks import CustomCallback

data_path = '../data/input/google-quest-challenge/'
bert_path = '../data/pretraining_models/bert/'
save_bert_path = '../../outputs/models/bert/'
max_sequence_length = 512
tokenizer = BertTokenizer.from_pretrained(bert_path + 'bert-base-uncased-tokenizer/vocab.txt')


def train():
    df_train = pd.read_csv(data_path + 'train.csv', header=0, encoding='utf-8')
    df_test = pd.read_csv(data_path + 'test.csv', header=0, encoding='utf-8')
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    print('\ntrain shape: {}'.format(df_train.shape))
    print('test shape: {}'.format(df_test.shape))
    print('df shape: {}'.format(df.shape))

    input_categories = list(df_train.columns[[1, 2, 5]])
    label_categories = list(df_train.columns[11:])
    print('input categories: \n\t{}\n'.format(input_categories))
    print('label categories:\n\t{}\n'.format(label_categories))

    # df_train, df_test, df = preprocessing(df_train, df_test, df, input_categories)

    inputs = compute_input_arrays(df_train, input_categories, tokenizer, max_sequence_length)
    test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, max_sequence_length)
    labels = compute_label_arrays(df_train, label_categories)

    cols_name_1 = [i for i in df.columns if i.startswith('host_') or i.startswith('cat_')]
    cols_name_2 = ['question_body_num_words', 'answer_num_words', 'question_vs_answer_length', 'q_a_author_same',
                   'answer_user_cat', 'indirect', 'question_count', 'reason_explanation_words', 'choice_words']
    # 将数据集划分为10等份，9份用于训练，1份用于测试。
    # 训练10次，得到10个结果，求均值作为最终的预测
    histories = []
    gkf = GroupKFold(n_splits=10).split(X=df_train.question_body, groups=df_train.question_body)
    for fold, (train_idx, valid_idx) in enumerate(gkf):
        if fold < 10:
            print(f"The fold-{fold} starts to run.")
            K.clear_session()
            model = create_bert_model()

            # 取出划分好的split fold数据
            train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]  # 分别取出ids, masks, segments
            train_labels = labels[train_idx]
            valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
            valid_labels = labels[valid_idx]

            # history = train(model=model,
            #                 train_data=(train_inputs, train_labels),
            #                 valid_data=(valid_inputs, valid_labels),
            #                 test_data=test_inputs,
            #                 learning_rate=3e-5, epochs=7, batch_size=8,
            #                 loss_function='binary_crossentropy', fold=fold)

            custom_callback = CustomCallback(
                valid_data=(valid_inputs,
                            # + [df_train[cols_name_1].iloc[valid_idx, :].to_numpy()]
                            # + [df_train[cols_name_2].iloc[valid_idx, :].to_numpy()],
                            valid_labels),
                test_data=test_inputs,  # + [df_test[cols_name_2].to_numpy()],
                save_model_path=save_bert_path,
                batch_size=8, fold=fold)

            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
            model.compile(loss='binary_crossentropy', optimizer=optimizer)
            model.fit(x=train_inputs,
                      # + [df_train[cols_name_1].iloc[train_idx, :].to_numpy()]
                      # + [df_train[cols_name_2].iloc[train_idx, :].to_numpy()],
                      y=train_labels, batch_size=8, epochs=5, callbacks=[custom_callback])

            histories.append(custom_callback)

    test_predictions = [histories[i].test_predictions for i in range(len(histories))]
    test_predictions = [np.average(test_predictions[i], axis=0) for i in range(len(test_predictions))]
    test_predictions = np.mean(test_predictions, axis=0)
    df_sub = pd.read_csv(data_path + 'sample_submission.csv')
    df_sub.iloc[:, 1:] = test_predictions

    df_sub.to_csv('submission-test.csv', index=False)


def main():
    set_gpu(gpu_id=-3)
    train()


if '__main__' == __name__:
    main()
