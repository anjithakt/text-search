# Assuming a 'term' to be a sequence of alphabets or numbers. Punctuations and white-spaces are ignored.
# The inverted index is a combination of three dictionaries:
#       doc[term] - A map from a term to the number of documents in which it occurs.
#       count[term] - A map from a term to its frequency in the file.
#       occur[term] - A map from a term to all the occurences of the term in format (document number, position/offset
#                                                                                            of term in the document)


import re
import sys
import operator
import random
from math import log2, sqrt
from collections import defaultdict


INF = 99999999
NEG_INF = -1 * INF

INF_POSITION = (INF, INF)
NEG_INF_POSITION = (NEG_INF, NEG_INF)

c = dict()
s = dict()


# Reads a file and returns it's content
def read_file(filename):
    handle = open(filename, encoding='utf8')
    data = handle.readlines()
    return data


# Constructs the inverted index.
def create_index(data):
    count = dict()
    occur = dict()
    doc = dict()
    documents = dict()

    doc_num = 1
    word_counter = 0

    for line in data:
        # An empty line (\n\n) indicates a new document, so we update the current document number.
        if (len(line.strip()) < 1):
            doc_num = doc_num + 1
            word_counter = 0
            continue

        # Lines are split into words along white-spaces and punctuations.
        words = re.split('[^0-9a-z]+', line.lower())

        # Eliminates the empty words in 'words' formed when we re.split() lines that begin or end in punctuations/space.
        for word in words:
            if (word == ''):
                continue

            word_counter = word_counter + 1
            # the frequencies of each terms are added to 'count'.
            count[word] = count.get(word, 0) + 1
            if (word not in occur):
                occur[word] = []
            # all occurences of each term are added to 'occur'.
            occur[word].append((doc_num, word_counter))
            if (doc_num not in documents):
                documents[doc_num] = {}
            documents[doc_num][word] = documents[doc_num].get(word, 0) + 1

        for term, freq in sorted(count.items()):
            prev_doc = None
            doc_count = 0

            for doc_no, _ in occur[term]:
                # Counts the number of documents by incrementing doc_count if doc number has changed
                if (prev_doc != doc_no):
                    doc_count = doc_count + 1
                    prev_doc = doc_no
                    # the number of documents each term occurs in are added to 'doc'.
                doc[term] = doc_count

    return count, occur, doc, doc_num, documents


# Finds the first occurence of the term
def first(term, index):
    first_occur = index[term][0]
    return first_occur


# Finds the last occurence of the term
def last(term, index, freq):
    n = freq[term]
    last_occur = index[term][n - 1]
    return last_occur


# Binary searches between low and high to find the next occurence of term after "current"
def binarysearch(term, low, high, current, index):
    mid = int((low + high) / 2)
    if (index[term][mid] == current):
        return mid
    elif (mid == high and index[term][mid] < current):
        return mid + 1
    elif (mid == high and index[term][mid] > current):
        return mid
    elif (high < low):
        return low
    elif (index[term][mid] < current):
        return binarysearch(term, mid + 1, high, current, index)
    else:
        return binarysearch(term, low, mid - 1, current, index)


# Finds the next occurence of term after current. Uses galloping search algorithm from the slides/text.
def next(term, current, index, freq):
    # Returns infinity if the term doesn't occur at all or if the last occurence of the term was before "current".
    if (freq[term] == 0 or last(term, index, freq) <= current):
        return INF_POSITION

    # Returns first occurence of term if there were no occurences of the term before current.
    if (first(term, index) > current):
        c[term] = 0
        return first(term, index)

    if (c[term] > 0 and index[term][c[term] - 1] <= current):
        low = c[term] - 1
    else:
        low = 1

    jump = 1
    high = low + jump

    # Gallop searches to find a position in the index of term greater than current.
    while (high < freq[term] and index[term][high] <= current):
        low = high
        jump = 2 * jump
        high = low + jump

    if (high > freq[term]):
        high = freq[term]

    next_pos = binarysearch(term, low, high, current, index)
    c[term] = next_pos
    # If the position obtained is current itself, return the position where term occurs immediately after that.
    if (index[term][next_pos] == current):
        return index[term][next_pos + 1]
    else:
        return index[term][next_pos]


# Finds the previous occurence of term before current.  Uses galloping search algorithm.
def prev(term, current, index, freq):
    low = 1
    jump = 1
    high = low + jump
    # Returns negative infinity if the term doesn't occur at all or first occurence of the term is after current.
    if (freq[term] == 0 or first(term, index) >= current):
        return NEG_INF_POSITION
    # Returns the last occurence of term as prev(term,current) if current comes after the last occurence of term.
    elif (last(term, index, freq) < current):
        c[term] = freq[term] - 1
        return last(term, index, freq)
    # Gallop searches to find a position in the index of term greater than current.
    while (high < freq[term] and index[term][high] <= current):
        low = high
        jump = 2 * jump
        high = low + jump

    if (high > freq[term]):
        high = freq[term]

    pos = binarysearch(term, low, high, current, index)
    return index[term][pos - 1]


# A line is converted to a dictionary of its words and their frequencies, ignoring white spaces/punctuation.
def vectorize(line):
    query_vect = dict()
    # The line is converted to lowercase and split along whitespaces and special characters.
    words = re.split('[^0-9a-z]+', line.lower())
    for word in words:
        # Check for empty "words"
        if (len(word) < 1):
            continue
        # Finds the frequencies of each words in the line
        query_vect[word] = query_vect.get(word, 0) + 1
    return query_vect


# Finds the next document where the term appears after current.
def next_doc(term, current, index, freq):
    if (term not in index):
        return INF
    elif (current + 1, 0) in index[term]:
        return current + 1
    else:
        return next(term, (current + 1, 0), index, freq)[0]

# Calculates the TF-IDF component to calculate BM25 score
def bm25(term, docid, documents, doc_num, doc, l_avg):
    doc_vector = documents[docid]
    k1 = 1.2
    b = 0.75
    l_d = sum(doc_vector.values())
    f = doc_vector[term]
    # Applies the formula for TF
    TF = (f * (k1 + 1)) / (f + k1 * ((1 - b) + b * (l_d / l_avg)))
    # Finds IDF
    IDF = doc_num / doc[term]
    score = TF * IDF
    return score

# Ranks and returns the score of the documents using BM25 scoring with accumulator pruning. a_max and u are input by the user.
def rank_bm25(query, k, a_max, u, doc_num, doc, index, freq, documents):
    document_list = documents.values()
    l_sum = 0
    # l_avg is calculated to pass to the function which calculates BM25 score
    for each_document in document_list:
        l_sum = l_sum + sum(each_document.values())
    l_avg = l_sum / doc_num

    # Sorting terms in the query in increasing order of N_t/ doc[term]
    terms = [x for x in query.keys() if x in freq]
    terms = sorted(terms, key=lambda a: doc[a])

    # Initialize two empty accumulators of size a_max+1 (the extra 1 is for end of list marker)
    acc = [(None, 0)] * (a_max + 1)
    acc_length = 0
    acc_updated = [(None, 0)] * (a_max + 1)
    acc[0] = (INF, 0)
    for term in terms:
        inpos = 0
        outpos = 0
        quotaLeft = a_max - acc_length

        # Finds all the distinct documents in which a term occurs
        def distinct_documents(t):
            current = 0
            while True:
                next = next_doc(t, current, index, freq)
                if next != INF:
                    yield next
                    current = next
                else:
                    raise StopIteration
        # Updates the current position of acc_updated with contents of current position of acc.
        def copyNextAcc(inpos, outpos):
            acc_updated[outpos] = acc[inpos]
            return inpos+1, outpos + 1
        # Updates the score for the corresponding document.
        def addWithUpdate(doc_number, inpos, outpos):
            score = bm25(term, doc_number, documents, doc_num, doc, l_avg)
            acc_updated[outpos] = (doc_number, score)
            if acc[inpos][0] == doc_number:
                acc_updated[outpos] = (doc_number, acc_updated[outpos][1] + acc[inpos][1])
                inpos = inpos + 1
            outpos = outpos + 1
            return (inpos, outpos)
        # In case there is enough free accumulators no pruning required
        if (quotaLeft >= doc[term]):
            for docid in distinct_documents(term):
                while acc[inpos][0] is not None and acc[inpos][0] < docid:
                    inpos, outpos = copyNextAcc(inpos, outpos)
                inpos, outpos = addWithUpdate(docid, inpos, outpos)
            while acc[inpos][0] is not None and acc[inpos][0] < INF:
                inpos, outpos = copyNextAcc(inpos, outpos)
            acc_updated[outpos] = (INF, 0)
            acc,acc_updated = acc_updated, acc
            acc_length = outpos
        # In case accumulator limit has already reached, do not create new accumulators
        elif (quotaLeft <= 0):
            for i in range(acc_length):
                acc_updated[i] = (acc[i][0], acc[i][1] + bm25(term, acc[i][0], documents, doc_num, doc, l_avg))
            acc, acc_updated = acc_updated, acc
        # Pruning is required when there is no enough quote to create all new accumulators
        else:
            vtf = 1
            tf_stats = defaultdict(int)
            posting_seen = 0
            inpos = 0
            outpos = 0

            for docid in distinct_documents(term):
                while acc[inpos][0] is not None and acc[inpos][0] < docid:
                    inpos, outpos = copyNextAcc(inpos, outpos)
                if acc[inpos][0] == docid:
                    inpos, outpos = addWithUpdate(docid, inpos, outpos)
                elif quotaLeft > 0:
                    f_tid = documents[docid][term]
                    # Check if the frequency of term in the document is greater than the threshold.
                    if f_tid >= vtf:
                        inpos, outpos = addWithUpdate(docid, inpos, outpos)
                        quotaLeft -= 1
                    tf_stats[f_tid] = tf_stats[f_tid] + 1
                posting_seen += 1
                if posting_seen % u == 0:
                    q = (doc[term] - posting_seen) / posting_seen
                    tf_sum = 0
                    for (tf_num, stat) in sorted(tf_stats.items()):
                        tf_sum += stat * q
                        if tf_sum >= quotaLeft:
                            vtf = tf_num
                            break
            while acc[inpos][0] is not None and acc[inpos][0] < INF:
                inpos, outpos = copyNextAcc(inpos, outpos)
            acc_updated[outpos] = (INF, 0)
            acc,acc_updated = acc_updated, acc
            acc_length = outpos
    acc = [i for i in acc if i[1] > 0]
    acc = sorted(acc, key=lambda a: a[1], reverse=True)
    return acc[:k]


# Finds the the next cover of the query_vector after "current".
def next_cover(current, query_vector, index, freq):
    v = None
    # Finds the maximum of the next positions after current of all the terms in the query.
    for term in query_vector:
        pos1 = next(term, current, index, freq)
        if (not v or (pos1 > v)):
            v = pos1
    # If the maximum is infinity, there are no more covers and we return next_cover as infinity
    if (v == INF_POSITION):
        return (v, v)

    u = None
    # Finds the minimum of the previous positions of all the terms in the query, starting from v+1.
    for term in query_vector:
        pos2 = prev(term, (v[0], v[1] + 1), index, freq)
        if (not u or (pos2 < u)):
            u = pos2
    # If the positions u and v belong to the same document, we return (u,v) as the next cover.
    if (u[0] == v[0]):
        return (u, v)
    else:
        return next_cover(u, query_vector, index, freq)


# The documents are ranked based on proximity score (based on number and sizes of covers in the document)
def rank_proximity(query_vector, k, index, freq):
    scores = dict()
    u = NEG_INF_POSITION
    # Finds the first cover for the query.
    (u, v) = next_cover(u, query_vector, index, freq)
    d = u[0]
    score = 0

    while (u < INF_POSITION):
        # If the document has changed, assign the remembered score to the previous document.
        if (d < u[0]):
            scores[d] = score
            d = u[0]
            score = 0
        # The score is calculated for each document from the covers returned by next_cover.
        score = score + 1 / (v[1] - u[1] + 1)
        (u, v) = next_cover(u, query_vector, index, freq)
    # Assigns the score for the last document.
    if (d < INF):
        scores[d] = score
    # Ranks and returns the first k documents according to the score.
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_scores[:k]


# Gets the filename, rank_method,k and query from the command line.
# Assuming the code will be run from command line as : python3 SearchIndex.y some_file rank_method num_results query a_max u
filename = sys.argv[1]
rank_method = sys.argv[2]
k = int(sys.argv[3])
query_phrase = sys.argv[4]
a_max = int(sys.argv[5])
u = int(sys.argv[6])

ranked_list = list()

# Data is read from the file and the index is created.
data = read_file(filename)
count, occur, doc, doc_num, documents = create_index(data)

# Query vector is split into a dictionary of words and their frequencies (ignoring non alpha-numeric characters)
query_vector = vectorize(query_phrase)
q = random.randint(1,101)

# Check for empty query.
if (len(query_vector) < 1):
    print("The query is empty")
else:
    # Calls the appropriate ranking functions for the rank_method input by the user.
    if (rank_method.lower() == "bm25"):
        ranked_list = rank_bm25(query_vector, k,a_max,u,doc_num, doc, occur, count, documents)

    elif (rank_method.lower() == "proximity"):
        ranked_list = rank_proximity(query_vector, k, occur, count)
    else:
        print("Unknown Rank Method")
# The documents are their scores are output in the format of trec_top_file
if (len(ranked_list) > 0):
    for doc, score in ranked_list:
        print(str(q) + " 0 " + str(doc) + " 1" + " " + str(score) + " run-nam1")
