import re
from typing import List
from nltk import PorterStemmer
from krovetzstemmer import Stemmer as KrovetzStemmer

# Suffix-s
def suffix_stem(word, lower=True):
    if lower:
        word = word.lower()
    suffixes = ['s']
    for suff in suffixes:
        if word.endswith(suff):
            return word[:-len(suff)]

    return word

def stem_text(text, stemmer: str):
    '''stemmer could be either of ['suffix', 'porter', 'krovetz']'''

    stemmer_function = {
        'suffix':   suffix_stem,
        'porter':   PorterStemmer().stem,
        'korvetz':  KrovetzStemmer().stem,
    }

    if not stemmer in stemmer_function:
        raise ValueError(
            f'stemmer argument mustbe either of {list(stemmer_function.keys())}')
    
    return ' '.join(stemmer_function[stemmer](word) for word in text.split())

def analyze_text_with_all_stemmers(text):
    # Suffix-s
    print("Suffix-s")
    print(stem_text(text, 'suffix'))
    print("\n")

    # Porter
    print("Porter")
    ps = PorterStemmer()
    print(stem_text(text, 'porter'))
    print("\n")

    # Krovetz
    print("Krovetz")
    ks = KrovetzStemmer()
    print(stem_text(text, 'korvetz'))
    print("\n")

def main():
    text = """Two fathers and two sons went to fish. 
    They saw a tall and strong tree that reached into the 
    heaven for all the world to see they 
    spend three hours there before they headed home"""

    analyze_text_with_all_stemmers(text)



if __name__ == '__main__':
    main()