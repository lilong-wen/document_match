import sys
import copy
import operator

def invertedIndex(corpus):
    dictionary = []
    inverted_index = dict()
    with open(corpus) as corpus:
        doc_lines = [line.strip('\n') for line in corpus]
    total_documents = len(doc_lines)
    for line in doc_lines:
        temp_dict = dict()
        doc_line = line.split('\t')
        temp_dict[doc_line[0]] = doc_line[1]
        dictionary.append(temp_dict)
    dictionary_sorted = sorted(dictionary, key=lambda temp_dict: list(temp_dict.keys())[0])
    for temp_dict in dictionary_sorted:
        inverted_index_freq = {}
        doc_id = list(temp_dict.keys())[0]
        temp_line = temp_dict[doc_id]
        token_gen = temp_line.split(' ')
        for token in token_gen:
            if token in inverted_index:
                if doc_id not in inverted_index[token]:
                    inverted_index[token].append(doc_id)
            else:
                inverted_index[token] = [doc_id]
    return inverted_index,dictionary_sorted

def printInvertedIndex(inverted_index):
    for key in inverted_index.keys():
        print(key)
        print(inverted_index[key])
    print('Number of tokens in dictionary:: ' + str(len(inverted_index.keys())))

def daat(inFile, outFile, inverted_index, sorted_dictionary):
    with open(inFile) as query_doc:
        query_lines = [line.strip('\n') for line in query_doc]
    query_tokens = []
    tokens_per_query = []
    count = 0
    for line in range(0,len(query_lines)):
        count = count+1
        query_line = query_lines[line].split(' ')
        query_tokens.append(query_line)
        if (line != 0):
            print("INSIDE LINE 0")
            file1 = open(outFile, "a+")
            file1.write("\n\n")
            file1.close()
        calculateAND_OR(query_line, inverted_index, outFile, sorted_dictionary)
    for i in range(count):
        tokens_per_query.append(len(query_tokens[i]))

def getPostingListandPrint(inverted_index_and, inverted_index_or, key, posting_list_or, posting_list_and):
    posting_list_and.append(inverted_index_and[key])
    posting_list_or.append(inverted_index_or[key])
    with open(outFile, "a") as out:
        out.write("GetPostings" + "\n")
        out.write(key + "\n")
        out.write("Posting List: " + ' '.join(inverted_index[key]) + "\n")
    return posting_list_and, posting_list_or

def calc_tfidf(query_line, result_and, inverted_index, sorted_dictionary):
    total_documents = int(len(sorted_dictionary.keys()))
    result_dict = dict()
    for list in result_and:
        totaltdf = 0
        for token in query_line:
            tf = sorted_dictionary[list].split(' ').count(token) / len(sorted_dictionary[list].split(' '))
            idf = total_documents / len(inverted_index[token])
            totaltdf += idf * tf
        result_dict[list] = totaltdf
    final = sorted(result_dict.items(), key=operator.itemgetter(1))
    list = []
    for doc_id in reversed(final):
        list.append(doc_id[0])
    return list

def calculateAND_OR(query_line, inverted_index, outFile, sorted_dictionary):
    inverted_index_or = copy.deepcopy(inverted_index)
    inverted_index_and = copy.deepcopy(inverted_index)
    posting_list_and = []
    posting_list_or = []
    for key in query_line:
        temp_posting_and, temp_posting_or = getPostingListandPrint(inverted_index_and, inverted_index_or, key, posting_list_or, posting_list_and)
    posting_list_and = copy.deepcopy(temp_posting_and)
    posting_list_or = copy.deepcopy(temp_posting_or)

    # DAAT AND FUNCTION
    no_list_empty = True
    comparison_and = 0
    result_and = []
    while no_list_empty:
        number_of_list_having_min = 0
        min_val = 99999999
        for i in range(0, len(posting_list_and)):
            if i == 0:
                min_val = posting_list_and[i][0]
            else:
                if int(min_val) > int(posting_list_and[int(i)][0]):
                    min_val = posting_list_and[i][0]
        for list in posting_list_and:
            if (list[0] == min_val):
                number_of_list_having_min += 1
        comparison_and = comparison_and + (len(posting_list_and) - 1)
        if number_of_list_having_min == len(posting_list_and):
            result_and.append(posting_list_and[0][0])
            for list in posting_list_and:
                list.pop(0)
        else:
            for list in posting_list_and:
                if list[0] == min_val:
                    list.pop(0)
        for list in posting_list_and:
            if len(list) == 0:
                no_list_empty = False


    # DAAT OR FUNCTION
    result_or = []
    comparison_or = 0
    exit_condition = True
    while exit_condition:
        number_of_nonempty = 0
        for list in posting_list_or:
            if len(list) > 0:
                number_of_nonempty = number_of_nonempty + 1
        if number_of_nonempty <= 1:
            exit_condition = False
            for list in posting_list_or:
                if (len(list) > 0):
                    result_or.extend(list)
        else:
            minimum_value = 99999999999999
            for list_index in range(0,len(posting_list_or)):
                if(len(posting_list_or[list_index]) > 0):
                    if ((int(posting_list_or[int(list_index)][0]) < int(minimum_value))):
                        minimum_value = posting_list_or[int(list_index)][0]
            result_or.append(minimum_value)
            for list in posting_list_or:
                if len(list) > 0:
                    if list[0] == minimum_value:
                        comparison_or += 1
                        list.pop(0)
                    else:
                        comparison_or += 1
            comparison_or -= 1


    # OUTPUT TO FILE
    with open(outFile, "a") as out:
        out.write("DaatAnd" + "\n")
        out.write(' '.join(query_line))
        out.write("\n")
        if len(result_and) == 0:
            out.write("Results: empty" + "\n")
        else:
            out.write("Results: " + ' '.join(result_and) + "\n")
        out.write("Number of documents in results: " + str(len(result_and)) + "\n")
        out.write("Number of comparisons: " + str(comparison_and) + "\n")
        out.write("TF-IDF\n")
        result_weighted = calc_tfidf(query_line, result_and, inverted_index, sorted_dictionary)
        if len(result_weighted) == 0:
            out.write("Results: empty" + "\n")
        else:
            out.write("Results: "+' '.join(result_weighted)+"\n")
        out.write("DaatOr" + "\n")
        out.write(' '.join(query_line))
        out.write("\n")
        if len(result_or) == 0:
            out.write("Results: empty")
        else:
            out.write("Results: " + ' '.join(result_or) + "\n")
        out.write("Number of documents in results: " + str(len(result_or)) + "\n")
        out.write("Number of comparisons: " + str(comparison_or)+"\n")
        out.write("TF-IDF\n")
        result_weighted = calc_tfidf(query_line, result_or, inverted_index, sorted_dictionary)
        if len(result_weighted) == 0:
            out.write("Results: empty" + "\n")
        else:
            out.write("Results: " + ' '.join(result_weighted))

def findMinimumLengthPosting(posting_list):
    min_len = 999999
    for i in range(len(posting_list)-1):
        p1 = posting_list[i]
        p2 = posting_list[i + 1]
        if(len(p1) < min_len and len(p1) < len(p2)):
            min_list = p1
            min_len = len(p1)
        elif(len(p2) < len(p1) and len(p2) < min_len):
            min_list = p2
            min_len = len(p2)
    return min_list

if __name__ == "__main__":
    corpus = str(sys.argv[1])
    outFile = str(sys.argv[2])
    inFile = str(sys.argv[3])
    inverted_index,sorted_dictionary = invertedIndex(corpus)
    main_dict = dict()
    for i in sorted_dictionary:
        main_dict.update(i)
    # printInvertedIndex(inverted_index)
    daat(inFile, outFile, inverted_index, main_dict)
