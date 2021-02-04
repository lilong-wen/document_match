import operator
import math
import time
import argparse
from tfidf import Reader, TF_IDF

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_path", default='../data/scanned_result_small.json')
    parser.add_argument("--query_path", default='../data/camera_result_small.json')
    args = parser.parse_args()

    read = Reader()
    content_data = read.read_data(args.content_path)
    query_data = read.read_data(args.query_path)
    tf_idf = TF_IDF()

    content_processed_data = tf_idf.pre_process(content_data)
    query_processed_data = tf_idf.pre_process(query_data)
    inverted_index = tf_idf.generate_inverted_index(content_processed_data)
    idf_scores_content = tf_idf.calculate_idf(content_processed_data)
    tf_idf_scores_content = tf_idf.calculate_tfidf(content_processed_data,\
                                                   idf_scores_content)
    tf_idf_scores_query = tf_idf.calculate_tfidf(query_processed_data,\
                                                   idf_scores_content)
    query_docs = {}

    for q_doc_num, tokens in query_processed_data.items():

        time_start = time.time()
        doc_sim = {}

        for token_item in tokens:

            if token_item in inverted_index.keys():
                doc_nums = inverted_index[token_item]
                for doc_item in doc_nums:
                    content_doc_score = tf_idf_scores_content[doc_item][token_item]
                    content_doc_length = math.sqrt(sum(x ** 2 for x in \
                                                       tf_idf_scores_content[doc_item].values()))
                    query_doc_score = tf_idf_scores_query[q_doc_num][token_item]
                    query_doc_length = math.sqrt(sum(x ** 2 for x in \
                                                     tf_idf_scores_query[q_doc_num].values()))

                    cosine_sim = (content_doc_score * query_doc_score) / \
                        (content_doc_length * query_doc_length)

                    if doc_item in doc_sim.keys():
                        doc_sim[doc_item] += cosine_sim
                    else:
                        doc_sim[doc_item] = cosine_sim

        ranked = sorted(doc_sim.items(), key=operator.itemgetter(1), \
                        reverse=True)
        query_docs[q_doc_num] = ranked
        print(q_doc_num)

        print(query_docs)

        time_end = time.time()
        # print('time cost ', time_end - time_start, 's')

    count = 0
    total_num = len(query_docs)
    for doc_num, value in query_docs.items():
        matched_id = value[0][0]
        if int(doc_num) != int(matched_id):
            print(f"{doc_num} mismatched with {matched_id}")
        else:
            count += 1

    print(f"acc is {count / total_num:.4f}")
