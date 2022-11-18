from collections import Counter, defaultdict
from typing import Dict, List
from stemmers import stem_text, remove_punctuation, remove_stopwords

def index():
    pass

def print_pretty_index(index: Dict[str, List]):
    pretty_index = []
    for word, postings in index.items():
        pretty_postings = f'{word}->'
        for posting in postings:
            pretty_postings += f'{posting[0]}:{posting[1]}, '
        pretty_index.append(pretty_postings[:-2])

    for line in pretty_index:
        print(line)

        

def main():
    docs = [
        'The old man and his two sons went fishing.',
        'Recreational fishing is an activity with important social implications.',
        'Introduction to social protection benefits for old age',
        'Introduction to how lake trout fishing works.'
    ]

    stopwords = ['an', 'and', 'are', 'for', 'how', 'in', 'is',
                 'not', 'or', 'the', 'these', 'this', 'to', 'with']

    pipeline = [
        remove_punctuation,
        str.lower,
        lambda t: remove_stopwords(t, stopwords),
        lambda t: stem_text(t, 'suffix'),
        str.split,
    ]

    tokenized_docs = []
    for d in docs:
        for p in pipeline:
            d = p(d)
        tokenized_docs.append(d)
    
    
    index = defaultdict(list)
    vocab = sorted(list(set(token for tokens in tokenized_docs for token in tokens)))

    for token in vocab:
        for i, tokens in enumerate(tokenized_docs):
            n_token_in_doc = Counter(tokens)[token]
            if n_token_in_doc > 0:
                # DOC ID STARTS AT 1!
                doc_id = i + 1
                # TODO: ADD THE TYPE OF PAYLOAD YOU WANT HERE!!!
                payload = n_token_in_doc
                posting = (doc_id, payload)
                index[token].append(posting)  # document (index + 1) is added to index
            
    print_pretty_index(index)




if __name__ == '__main__':
    main()