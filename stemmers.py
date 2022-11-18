from nltk import PorterStemmer
from krovetzstemmer import Stemmer as KrovetzStemmer

SUFFIXES = ['ing', 's']

# Suffix-s
def suffix_stem(word, lower=True):
    if lower:
        word = word.lower()
    for suff in SUFFIXES:
        if word.endswith(suff):
            return word[:-len(suff)]

    return word


def stem_text(text, stemmer_func):
    return ' '.join(stemmer_func(word) for word in text.split())

def main():
    text = """
    Two fathers and two sons went to fish. They saw a tall and strong tree that reached into the 
    heaven for all the world to see they spend three hours there before they headed home
    """

    # Suffix-s
    print("Suffix-s")
    print(stem_text(text, suffix_stem))
    print("\n")

    # Porter
    print("Porter")
    ps = PorterStemmer()
    print(stem_text(text, ps.stem))
    print("\n")

    # Krovetz
    print("Krovetz")
    ks = KrovetzStemmer()
    print(stem_text(text, ks.stem))
    print("\n")

if __name__ == '__main__':
    main()