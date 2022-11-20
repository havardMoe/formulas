
from ast import List
from typing import Iterable


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
