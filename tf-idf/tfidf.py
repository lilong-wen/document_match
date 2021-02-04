import string
import math
import nltk
import jieba
import json
import codecs

class TF_IDF:

    def __init__(self):
        pass

    def tokenize(self, origin_s):

        '''
        return: tokenized string
        '''

        translator = str.maketrans('', '', string.punctuation)
        refined_str = origin_s.translate(translator)
        cutted_str = jieba.cut(refined_str)
        tokenized_str = [item for item in cutted_str if item is not ' ']

        return tokenized_str

    def pre_process(self, contents):

        '''
        return: {doc_num1: tokenized_str1, doc_num2: tokenized_str2, ...}
        '''

        data_dict = {}

        for content_item in contents:
            tokens = self.tokenize(content_item[1])
            data_dict[content_item[0]] = tokens

        return data_dict

    def get_vocabulary(self, all_data):

        tokens = []

        for token_list in all_data.values():
            tokens = tokens + token_list

        fdist = nltk.FreqDist(tokens)

        return list(fdist.keys())

    def generate_inverted_index(self, data):

        '''
        return: {term1: [doc_num1, doc_num3,...], term2: [doc_num3,..], ...}
        '''

        all_words = self.get_vocabulary(data)
        index = {}

        for word in all_words:
            for doc_num, tokens in data.items():
                if word in index.keys():
                    index[word].append(doc_num)
                else:
                    index[word] = [doc_num]

        return index

    def calculate_idf(self, data):

        '''
        return: {word1: idf_score1, word2, idf_score2, ...}
        '''
        idf_score = {}

        data_len = len(data)
        all_words = self.get_vocabulary(data)

        for word in all_words:
            word_count = 0
            for token_list in data.values():
                if word in token_list:
                    word_count += 1
            idf_score[word] = math.log10(data_len / word_count)

        return idf_score

    def calculate_tf(self, tokens):

        tf_scores = {}

        for token_item in tokens:
            tf_scores[token_item] = tokens.count(token_item)

        return tf_scores

    def calculate_tfidf(self, data, idf_scores):

        '''
        return: {doc_num1: {token1: score, token2: score,...}, \
                 doc_num2: {token1: score, token2: score,...}, ...}
        '''

        tf_idf_scores = {}
        tf_scores = {}
        tf_idf_token_score = {}

        for doc_num, tokens in data.items():
            tf_scores[doc_num] = self.calculate_tf(tokens)

        for doc_num, tf_score_item in tf_scores.items():
            for token, score in tf_score_item.items():
                tf = score
                if token in idf_scores.keys():
                    idf = idf_scores[token]
                else:
                    idf = 0
                tf_idf_token_score[token] = tf * idf

            tf_idf_scores[doc_num] = tf_idf_token_score

        return tf_idf_scores


class Reader:

    def __init__(self):
        pass

    def read_data(self, path):

        '''
        return: [(doc_num1, string1), (doc_num2, string2), ...]
        '''

        with codecs.open(path, 'r', encoding='utf-8') as f_data:
            result_dict = json.load(f_data, encoding='utf-8')

        contents = []

        for item in result_dict.items():
            contents.append((int(item[0]), " ".join(item[1])))

        return contents

    def save_dict(self, path):
        pass

    def read_dict(self, path):
        pass
