import os
import operator
import math
import json
import argparse
import codecs
import string
import nltk
import jieba
import time

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
    # modified_string = s_data
    # for chinese string
    modified_string = jieba.cut(modified_string)
    tokenized_str = [item for item in modified_string if item is not " "]
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

    for token in tokens:
        tf_scores[token] = tokens.count(token)
        tf_scores[token] = 1

    return tf_scores

def calculate_tf_q(tokens):

    tf_scores = {}

    for token in tokens:
        tf_scores[token] = 1

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
            tf_score[token] = tf * idf
        tf_idf_scores[str(doc_num)] = tf_score

    return tf_idf_scores

def calculate_tfidf_queries(queries, idf_score):

    q_tf_idf_scores = {}
    tf_scores = {}

    for key, value in queries.items():
        tf_scores[key] = calculate_tf_q(value)

    for key, tf_score in tf_scores.items():
        for token, score in tf_score.items():
            tf = score
            if token in idf_score.keys():
                idf = idf_score[token]
            else:
                idf = 0
            tf_score[token] = tf * idf
        q_tf_idf_scores[str(key)] = tf_score

    return q_tf_idf_scores

def save_dict(dict_data, name):

    with codecs.open(name + ".json", 'w', encoding='utf-8') as f:
        json.dump(dict_data, f, ensure_ascii=False)

def read_dict(name):

    with codecs.open(name, 'r', encoding='utf-8') as f1:
        result_dict = json.load(f1, encoding='utf-8')

    return result_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_path", default='../data/scanned_result.json')
    parser.add_argument("--query_path", default='../data/camera_result.json')
    args = parser.parse_args()

    pre_processed_data = pre_process(read_data(args.content_path))

    queries = pre_process(read_data(args.query_path))

    # inverted_index = generate_inverted_index(pre_processed_data)
    # idf_scores = calculate_idf(pre_processed_data)
    # scores = calculate_tfidf(pre_processed_data, idf_scores)


    if os.path.exists("inverted_index.json"):
        print("load inverted index")
        inverted_index = read_dict("inverted_index.json")
    else:
        inverted_index = generate_inverted_index(pre_processed_data)
        save_dict(inverted_index, "inverted_index")

    if os.path.exists("idf_scores.json"):
        print("load idf scores")
        idf_scores = read_dict("idf_scores.json")
    else:
        idf_scores = calculate_idf(pre_processed_data)
        save_dict(idf_scores, "idf_scores")

    if os.path.exists("scores.json"):
        print("load tf-idf scores")
        scores = read_dict("scores.json")
    else:
        scores = calculate_tfidf(pre_processed_data, idf_scores)
        save_dict(scores, "scores")

    query_scores = calculate_tfidf_queries(queries, idf_scores)
    # query_scores_sorted = sorted(query_scores.items(), key=operator.itemgetter(1), reverse=True)

    query_docs = {}
    for key, value in queries.items():
        time_start=time.time()
        doc_sim = {}

        for term in value:
            if term in inverted_index.keys():
                docs = inverted_index[term]
                for doc in docs:
                    # doc = str(doc)
                    doc_score = scores[str(doc)][term]
                    doc_length = math.sqrt(sum(x ** 2 for x in scores[str(doc)].values()))
                    query_score = query_scores[str(key)][term]
                    query_length = math.sqrt(sum(x ** 2 for x in query_scores[str(key)].values()))
                    cosine_sim = (doc_score * query_score) / (doc_length * query_length)
                    if doc in doc_sim.keys():
                        doc_sim[doc] += cosine_sim
                    else:
                        doc_sim[doc] = cosine_sim

                    '''
                    if str(doc) == '187':
                        print(f'query doc num: {key}')
                        print(f'term: {term}', end='  |')
                        print(f'doc_num - doc_score: {doc} - {doc_score}', end=' | ')
                        print(f'query_score: {query_score}', end='  |')
                        print(f'cosine_sim: {cosine_sim}', end='  |')
                        print()
                    '''

        ranked = sorted(doc_sim.items(), key=operator.itemgetter(1), reverse=True)

        query_docs[key] = ranked
        time_end=time.time()
        print('time cost',time_end-time_start,'s')

    #print(query_docs)
    count = 0
    total_num = len(query_docs)
    for key, value in query_docs.items():
        matched_id = value[0][0]
        if int(key) != int(matched_id):
            print(f"{key} mismatched with {matched_id}")
        else:
            count += 1

    print(f"acc is {count/total_num:.3f}")
