import math
from typing import Iterable, List
import numpy as np

def jaccard_similarity_termbased(d1: Iterable[str], d2: Iterable[str]):
    '''
    Args:
        2 iterables with words, e.g. ['my', 'doc'] and ['second', 'doc']
    '''
    intersection = set(d1).intersection(set(d2))
    union = set(d1).union(set(d2))
    return len(intersection) / len(union)

def jaccard_similarity_vectorbased(v1: List[int], v2: List[int]):
    '''
    Args:
        2 vectors with couunt (or binary[0,1]) vector representation of words,
        e.g. [2, 1, 0, 0, 1, 1] and [1, 1, 0, 0, 0, 1]
    '''
    assert len(v1) == len(v2)
    n_words_present = 0
    sim_count = 0
    
    for wc1, wc2 in zip(v1, v2):
        if wc1 * wc2 > 0:
            sim_count += 1
        # Either of the counts are above 0
        if wc1 > 0 or wc2 > 0:
            n_words_present += 1
    
    return sim_count / n_words_present

def cosine_similarity(v1: List[int], v2: List[int]):
    '''
   Args:
        2 vectors with couunt (or binary[0,1]) vector representation of words,
        e.g. [2, 1, 0, 0, 1, 1] and [1, 1, 0, 0, 0, 1] 
    '''
    assert len(v1) == len(v2)
    dot_product = 0
    norm1 = 0
    norm2 = 0
    for wc1, wc2 in zip(v1, v2):
        dot_product += wc1 * wc2
        norm1 += wc1**2
        norm2 += wc2**2

    denom = norm1 * norm2

    return dot_product / np.sqrt(denom) if denom > 0 else 0

def get_tf_vector(doc_term_vector: List[int]) -> List[float]:    
    """Computes the normalized term frequency vector from a raw term-frequency vector."""
    sum_freq = sum(doc_term_vector)
    if sum_freq == 0:  # This would mean that the document has no content.
        return None    
    tf_vector = [freq / sum_freq for freq in doc_term_vector]
    return tf_vector

def get_term_idf(doc_term_matrix: List[List[int]], term_index: int) -> float:
    """Computes the IDF value of a term, given by its index, based on a document-term matrix."""
    N = len(doc_term_matrix)
    n_t = sum([1 if doc_freqs[term_index] > 0 else 0 for doc_freqs in doc_term_matrix])
    return math.log2((N + 1) / n_t)

def get_tfidf_vector(doc_term_matrix: List[List[int]], doc_index: int) -> List[float]:
    """Computes the TFIDF vector from a raw term-frequency vector."""
    tf_vector = get_tf_vector(doc_term_matrix[doc_index])
    tfidf_vector = []
    for term_index, tf in enumerate(tf_vector):
        idf = get_term_idf(doc_term_matrix, term_index)
        tfidf_vector.append(tf * idf)
    return tfidf_vector

def main():
    # # 2020 exam
    # d1 = [0, 5, 6 ,0, 1]
    # d2 = [1, 6, 3 ,3, 5]
    # d3 = [7, 0, 3 ,8, 0]
    # d4 = [4, 4, 9 ,0, 2]

    # docs = [d1, d2, d3, d4]

    # sim1 = cosine_similarity(d2, d4)
    # print(sim1)

    # tfidfvec1 = get_tfidf_vector(docs, 1)
    # print(tfidfvec1)

    # tfidfvec2 = get_tfidf_vector(docs, 3)
    # print(tfidfvec2)

    # sim = cosine_similarity(tfidfvec1, tfidfvec2)
    # print(sim)

    # 2021 exam
    d1 = 'this is the first text'
    d2 = 'this is the second text'
    print(jaccard_similarity_termbased(d1.split(), d2.split()))

    d1 = 'this is another first text'
    d2 = 'this is another strage second text'
    print(jaccard_similarity_termbased(d1.split(), d2.split()))






if __name__ == '__main__':
    main()