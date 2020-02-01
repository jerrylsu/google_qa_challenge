import numpy as np
import tensorflow as tf
from nltk.tokenize.treebank import TreebankWordTokenizer
from spacy.lang.en import English
import pandas as pd

nlp = English()
sentencizer = nlp.create_pipe('sentencizer')
nlp.add_pipe(sentencizer)

tokenizer_ = tf.keras.preprocessing.text.Tokenizer(lower=True)
tree_tokenizer = TreebankWordTokenizer()


def get_tree_tokens(s):
    return ' '.join(tree_tokenizer.tokenize(s))


def add_question_metadata_features(text):
    doc = nlp(text)
    indirect = 0
    choice_words = 0
    reason_explanation_words = 0
    question_count = 0

    for sent in doc.sents:
        if '?' in sent.text and '?' == sent.text[-1]:
            question_count += 1
            for token in sent:
                if token.text.lower() == 'why':
                    reason_explanation_words += 1
                elif token.text.lower() == 'or':
                    choice_words += 1
    if question_count == 0:
        indirect += 1

    return np.array([indirect, question_count, reason_explanation_words, choice_words])


def question_answer_author_same(df):
    q_username = df['question_user_name']
    a_username = df['answer_user_name']
    author_same = []

    for i in range(len(df)):
        if q_username[i] == a_username[i]:
            author_same.append(int(1))
        else:
            author_same.append(int(0))
    return author_same


def add_question_metadata_features(text):
    doc = nlp(text)
    indirect = 0
    choice_words = 0
    reason_explanation_words = 0
    question_count = 0

    for sent in doc.sents:
        if '?' in sent.text and '?' == sent.text[-1]:
            question_count += 1
            for token in sent:
                if token.text.lower() == 'why':
                    reason_explanation_words += 1
                elif token.text.lower() == 'or':
                    choice_words += 1
    if question_count == 0:
        indirect += 1

    return np.array([indirect, question_count, reason_explanation_words, choice_words])


def add_external_features(df, ans_user_and_category):
    df['question_body'] = df['question_body'].apply(lambda x: str(x))
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.count.html#pandas.Series.str.count
    df['question_body_num_words'] = df['question_body'].str.count('\S+')

    df['answer'] = df['answer'].apply(lambda x: str(x))
    df['answer_num_words'] = df['answer'].str.count('\S+')

    df['question_vs_answer_length'] = df['question_body_num_words'] / df['answer_num_words']

    df['q_a_author_same'] = question_answer_author_same(df)

    answer_user_cat = []
    for i in df[['answer_user_name', 'category']].values:
        if i in ans_user_and_category:
            answer_user_cat.append(int(1))
        else:
            answer_user_cat.append(int(0))
    df['answer_user_cat'] = answer_user_cat

    handmade_features = []
    for text in df['question_body'].values:
        handmade_features.append(add_question_metadata_features(text))

    return df, np.array(handmade_features)


def split_document(texts):
    """transform the graph to sentences.
    arg:
    texts: str

    return: list[list]
    """
    all_sents = []
    max_num_sentences = 0.0
    for text in texts:
        doc = nlp(text)
        sents = []
        for idx, sent in enumerate(doc.sents):  # split('. '), return sentences
            sents.append(sent.text)
        all_sents.append(sents)

    return all_sents


def preprocessing(df_train, df_test, df, input_categories):
    for col in input_categories:
        df_train[f'treated_{col}'] = df_train[col].apply(lambda s: get_tree_tokens(s))
        df_test[f'treated_{col}'] = df_test[col].apply(lambda s: get_tree_tokens(s))
        df[f'treated_{col}'] = df_test[col].apply(lambda s: get_tree_tokens(s))

    # X_train_question = df_train['question_body']
    # X_train_title = df_train['question_title']
    # X_train_answer = df_train['answer']
    #
    # X_test_question = df_test['question_body']
    # X_test_title = df_test['question_title']
    # X_test_answer = df_test['answer']
    #
    # tokenizer_.fit_on_texts(list(X_train_title) + list(X_train_question) +
    #                         list(X_train_answer) + list(X_test_title) +
    #                         list(X_test_question) + list(X_test_answer))
    #
    # # list[list]: question -> sentences, answer -> sentences
    # X_train_question = split_document(X_train_question)
    # # X_train_title = split_document(X_train_title)
    # X_train_answer = split_document(X_train_answer)
    # X_test_question = split_document(X_test_question)
    # X_test_answer = split_document(X_test_answer)

    ans_user_and_category = df_train[df_train[['answer_user_name', 'category']].duplicated(keep='first')][
        ['answer_user_name', 'category']].values

    df_train, handmade_features = add_external_features(df_train, ans_user_and_category)
    df_test, handmade_features_test = add_external_features(df_test, ans_user_and_category)

    df_train = pd.concat([df_train, pd.DataFrame(handmade_features,
                                                 columns=['indirect', 'question_count', 'reason_explanation_words',
                                                          'choice_words'])], axis=1)
    df_test = pd.concat([df_test, pd.DataFrame(handmade_features_test,
                                               columns=['indirect', 'question_count', 'reason_explanation_words',
                                                        'choice_words'])], axis=1)

    # 归一化
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    num_words_scaler = MinMaxScaler()
    df_train[['question_body_num_words', 'answer_num_words']] = num_words_scaler.fit_transform(
        df_train[['question_body_num_words', 'answer_num_words']].values)
    df_test[['question_body_num_words', 'answer_num_words']] = num_words_scaler.transform(
        df_test[['question_body_num_words', 'answer_num_words']].values)

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
    df = pd.concat([df, pd.get_dummies(df['host'], drop_first=False, prefix='host', dtype=float)], axis=1)
    df = pd.concat([df, pd.get_dummies(df['category'], drop_first=False, prefix='cat', dtype=float)], axis=1)

    df_train = pd.merge(df_train,
                        df[['qa_id'] + [i for i in df.columns if i.startswith('host_') or i.startswith('cat_')]],
                        how='inner', on='qa_id')
    df_test = pd.merge(df_test,
                       df[['qa_id'] + [i for i in df.columns if i.startswith('host_') or i.startswith('cat_')]],
                       how='inner', on='qa_id')

    df_train.drop(['host', 'category'], inplace=True, axis=1)
    df_test.drop(['host', 'category'], inplace=True, axis=1)

    return df_train, df_test, df
