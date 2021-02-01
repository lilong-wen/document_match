import time
from scipy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
import json
import numpy as np
import jieba
from gensim import corpora
from gensim import models
from gensim import similarities
from collections import defaultdict

def tf_compare(query_dict, src_dict):

    query_list_index = []
    query_list_content = []
    for q_item in query_dict.items():
        q_item_jieba = jieba.cut(' '.join(q_item[1]))
        query_list_index.append(q_item[0])
        query_list_content.append(" ".join([i for i in q_item_jieba if i!=" "]))

    src_list_index = []
    src_list_content = []
    for s_item in src_dict.items():
        s_item_jieba = jieba.cut(' '.join(s_item[1]))
        src_list_index.append(s_item[0])
        src_list_content.append(" ".join([i for i in s_item_jieba if i!=" "]))

    documents = src_list_content
    texts = [[word for word in docu.split(" ")] for docu in documents]

    #print(texts)
    frequency = defaultdict(int)
    for t1 in texts:
        for t2 in t1:
            frequency[t2] += 1

    texts_refine = [[t2 for t2 in t1 if frequency[t2] > 0] for t1 in texts]

    dictionary = corpora.Dictionary(texts_refine)

    corpus = [dictionary.doc2bow(text)for text in texts_refine]

    tfidf = models.TfidfModel(corpus)
    featureNum=len(dictionary.token2id.keys())
    index=similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=featureNum)

    total_num = len(query_list_index)
    count = 0
    for query_id, query_item in zip(query_list_index, query_list_content):
        new_vec=dictionary.doc2bow(query_item.split(" "))
        #sims=index[tfidf[new_vec]]
        sims = index[new_vec]
        #print(sims)
        match_id = src_list_index[np.argmax(sims)]
        if query_id == match_id:
            count += 1
        else:
            print(str(query_id) + ": mismatched with " + str(match_id))

    acc = count / total_num
    print("acc is : " + str(acc))

def tfidf_similarity(query_dict, src_dict):
    query_list_index = []
    query_list_content = []
    for q_item in query_dict.items():
        q_item_jieba = jieba.cut(' '.join(q_item[1]))
        query_list_index.append(q_item[0])
        query_list_content.append(" ".join([i for i in q_item_jieba if i!=" "]))

    src_list_index = []
    src_list_content = []
    for s_item in src_dict.items():
        s_item_jieba = jieba.cut(' '.join(s_item[1]))
        src_list_index.append(s_item[0])
        src_list_content.append(" ".join([i for i in s_item_jieba if i!=" "]))


    cv = TfidfVectorizer(tokenizer=lambda s: s.split(" "))

    count = 0
    total_num = len(query_list_index)
    for query_id, item_q in zip(query_list_index, query_list_content):
        time_start=time.time()
        corpus = src_list_content + [item_q]
        vectors = cv.fit_transform(corpus).toarray()
        k_list = []
        for i in vectors[:-1]:
            k = np.dot(i, vectors[-1]) / (norm(i) * norm(vectors[-1]))
            k_list.append(k)
        match_id = src_list_index[k_list.index(max(k_list))]
        time_end=time.time()
        print('time cost',time_end-time_start,'s')

        if query_id == match_id:
            count += 1
        else:
            print(str(query_id) + ": mismatched with " + str(match_id))

    acc = count / total_num
    print("acc is : " + str(acc))



if __name__ == '__main__':

    camera_ocr_result = '../data/camera_result.json'
    scanned_ocr_result = '../data/scanned_result.json'

    with codecs.open(camera_ocr_result, 'r', encoding='utf-8') as f1:
        camera_result_dict = json.load(f1, encoding='utf-8')

    with open(scanned_ocr_result, 'r', encoding='utf-8') as f2:
        scanned_result_dict = json.load(f2, encoding='utf-8')

    #tf_compare(camera_result_dict, scanned_result_dict)
    tfidf_similarity(camera_result_dict, scanned_result_dict)
