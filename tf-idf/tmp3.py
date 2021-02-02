import math
import json
import argparse
import codecs
import string
import nltk
import jieba

def read_data(path):

    with codecs.open(path, 'r', encoding='utf-8') as f1:
        result_dict = json.load(f1, encoding='utf-8')

    contents = []
    for item in result_dict.items():
        contents.append((int(item[0]), " ".join(item[1])))

    return contents

def tokenize(s_data):

    translator = str.maketrans('', '', string.punctuation)
    modified_string = s_data.translate(translator)
    # for chinese string
    modified_string = jieba.cut(modified_string)
    tokenized_str = [item for item in modified_string]
    # for English
    #tokenized_str = nltk.word_tokenize(modified_string)

    return tokenized_str

def pre_process(contents):

    data_dict = {}

    for content_item in contents:
        tokens = tokenize(content_item[1])
        data_dict[content_item[0]] = tokens

    return data_dict

def get_vocabulary(data):

    tokens = []
    for token_list in data.values():
        tokens = tokens + token_list

    fdist = nltk.FreqDist(tokens)

    return list(fdist.keys())

def generate_inverted_index(data):

    all_words = get_vocabulary(data)

    index = {}

    for word in all_words:
        for doc_num, tokens in data.items():
            if word in tokens:
                if word in index.keys():
                    index[word].append(doc_num)
                else:
                    index[word] = [doc_num]

    return index

def calculate_idf(data):

    idf_score = {}

    data_len = len(data)
    all_words = get_vocabulary(data)
    for word in all_words:
        word_count = 0
        for token_list in data.values():
            if word in token_list:
                word_count += 1
        idf_score[word] = math.log10(data_len / word_count)

    return idf_score

def calculate_tf(tokens):

    tf_scores = {}

    for token in tonkes:
        tf_scores[token] = tokens.count(token)

    return tf_scores

def calculate_tfidf(data, idf_score):

    tf_idf_scores = {}
    tf_scores = {}

    for key, value in data.items():
        tf_scores[key] = calculate_tf(value)

    for doc_num, tf_score in tf_scores.items():
        for token, score in tf_score.items():
            tf = score
            idf = idf_score[token]
            tf_idf_scores[token] = tf * idf

    return tf_idf_scores

def calculate_tfidf_queries(queries, idf_score):

    q_tf_idf_scores = {}
    tf_scores = {}

    for key, value in queries.items():
        tf_scores[key] = calculate_tf(value)

    for key, tf_score in tf_scores.items():
        for token, score in tf_score.items():
            tf = score
            if token in idf_score.keys():
                idf = idf_score[token]
            else:
                idf = 0
            q_tf_idf_scores[token] = tf * idf

    return q_tf_idf_scores

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_path", default='../data/scanned_result.json')
    parser.add_argument("--query_path", default='../data/camera_result.json')
    args = parser.parse_args()

    pre_processed_data = pre_process(read_data(args.content_path))

    queries = pre_process(read_data(args.query_path))

    # print(pre_processed_data[0])
    # print(queries[0])

    inverted_index = generate_inverted_index(pre_processed_data)

    idf_scores = calculate_idf(pre_processed_data)
    scores = calculate_tfidf(pre_processed_data, idf_score)

    query_scores = calculate_tfidf_queries(queries, idf_score)

    for key, value in queries.items():

        doc_sim = {}

        for term in value:
            if term in inverted_index.keys():
                docs = inverted_index[term]
                print(docs)
                exit()
