import re
from typing import List


def remove_stopwords(text: str, stopwords: List[str]):
    stopwords = set(stopwords)
    words = [word for word in text.split() if word not in stopwords]
    return ' '.join(words)

def remove_punctuations(text: str, punctuation: List[str], rep=' '):
    for p in punctuation:
        text = text.replace(p, rep)
    return text

def remove_nonalphanumeric(text: str):
    '''using regex to remove all non-alphanumerical characters'''
    return re.sub(r'[\W_]', ' ', text)

def remove_short_words(text: str, max_word_length_removed: int):
    words = text.split()
    filtered_words = [word for word in words if len(word) > max_word_length_removed]
    return ' '.join(filtered_words)

