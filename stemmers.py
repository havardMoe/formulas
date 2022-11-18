from nltk import PorterStemmer
from krovetzstemmer import Stemmer as KrovetzStemmer

SUFFIXES = ['ing']

# Suffix-s
def suffix_stem(word):
    for suff in SUFFIXES:
        if word.endswith(suff):
            return word[:-len(suff)]

    return word


# Def stem-text

def main():
    text = 'utilities'

    # Suffix-s
    print("Suffix-s")
    print(suffix_stem(text))
    print("\n")

    # Porter
    print("Porter")
    ps = PorterStemmer()
    print(ps.stem(text))
    print("\n")

    # Krovetz
    print("Krovetz")
    ks = KrovetzStemmer()
    print(ks.stem(text))
    print("\n")

if __name__ == '__main__':
    main()