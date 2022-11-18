from collections import Counter, defaultdict
from typing import Dict, List
from stemmers import stem_text
from text_cleaning import (
    remove_nonalphanumeric, 
    remove_punctuations, 
    remove_stopwords, 
    remove_short_words
)
import string

def find_position(tokens, word):
    pos = []
    for i, token in enumerate(tokens):
        if token == word:
            pos.append(i+1)  # starts at 1 
    return pos

def print_pretty_index(index: Dict[str, List]):
    pretty_index = []
    for word, postings in index.items():
        pretty_postings = f'{word}->'
        for posting in postings:
            pretty_postings += f'{posting[0]}:{posting[1]}, '
        pretty_index.append(pretty_postings[:-2])

    for line in pretty_index:
        print(line)

def ex2020():
    docs = [
        "If you're going to San Francisco be sure to wear some flowers in your hair.",
        'My Lord Bassanio, since you have found Antonio, we two will leave you: but at dinner-time, I pray you, have in mind where we must meet.',
        "A federal judge's approval of T-Mobile's takeover of Sprint will test whether three giants will compete as aggressively for cellphone users as four unequal players once did.",
    ]

    
    stopwords = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 
        'it', 'in', 'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 
        'that', 'the', 'their', 'then', 'there', 'these', 'they', 'this', 
        'to', 'was', 'will', 'with'
    ]


    pipeline = [
        str.lower,
        remove_nonalphanumeric,
        lambda text: remove_stopwords(text, stopwords),
        lambda text: remove_short_words(text, max_word_length_removed=2),
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
        for i, doc_tokens in enumerate(tokenized_docs):
            # TODO: remove (+ 1) if docs start at 0!
            doc_id = i + 1

            n_token_in_doc = Counter(doc_tokens)[token]
            if n_token_in_doc > 0:


                # TODO: ADD THE TYPE OF PAYLOAD YOU WANT HERE!!!
                #       [others/none, 'word frequency', 'word position']
                payload_type = 'word position'
                
                # Example of frequency payload:
                if payload_type == 'word frequency':
                    payload = n_token_in_doc
                    posting = (doc_id, payload)
                    index[token].append(posting)

                # Example of position payload:
                # Cannot calculate position from doc_tokens as there are preprocessed
                # Instead need to calculate them on purely splitted and lowercase
                # tokens:
                if payload_type == 'word position':
                    original_text = docs[i]
                    position_minimal_processing = (
                            remove_nonalphanumeric(original_text.lower()))
                    positional_tokens = position_minimal_processing.split()
                    positions = find_position(positional_tokens, token)
                    if len(positions) == 0:
                        raise ValueError('There should be returned positions, since n_token_in_doc is > 0')

                    payloads = [(doc_id, pos) for pos in positions]
                    index[token].extend(payloads)
            
    print_pretty_index(index)

def ex2021():
    docs = [
        'The old man and his two sons went fishing.',
        'Recreational fishing is an activity with important social implications.',
        'Introduction to social protection benefits for old age',
        'Introduction to how lake trout fishing works.'
    ]

    stopwords = ['an', 'and', 'are', 'for', 'how', 'in', 'is',
                 'not', 'or', 'the', 'these', 'this', 'to', 'with']

    pipeline = [
        lambda text: remove_punctuations(text, string.punctuation),
        str.lower,
        lambda text: remove_stopwords(text, stopwords),
        lambda text: stem_text(text, 'suffix'),
        # function to split text into words, could use some more 
        # advanced function in case of HTML markup or similar
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
        for i, doc_tokens in enumerate(tokenized_docs):
            # TODO: remove (+ 1) if docs start at 0!
            doc_id = i + 1

            n_token_in_doc = Counter(doc_tokens)[token]
            if n_token_in_doc > 0:


                # TODO: ADD THE TYPE OF PAYLOAD YOU WANT HERE!!!
                #       [others/none, 'word frequency', 'word position']
                payload_type = 'word frequency'
                
                # Example of frequency payload:
                if payload_type == 'word frequency':
                    payload = n_token_in_doc
                    posting = (doc_id, payload)
                    index[token].append(posting)

                # Example of position payload:
                # Cannot calculate position from doc_tokens as there are preprocessed
                # Instead need to calculate them on purely splitted and lowercase
                # tokens:
                if payload_type == 'word position':
                    original_text = docs[i]
                    position_minimal_processing = (
                            remove_nonalphanumeric(original_text.lower()))
                    positional_tokens = position_minimal_processing.split()
                    positions = find_position(positional_tokens, token)
                    if len(positions) == 0:
                        raise ValueError('There should be returned positions, since n_token_in_doc is > 0')

                    payloads = [(doc_id, pos) for pos in positions]
                    index[token].extend(payloads)
            
    print_pretty_index(index)

def ex2022():
    # TODO: Insert documents here
    docs = [
        ... ,
        ...,
        ...,
        ...
    ]

    # TODO: If you are to use stopwords for the task: insert them here
    # Note: see ex2020 and ex2021 for examples of stopwords if none is specified
    stopwords = ['an', 'and', 'are', 'for', 'how', 'in', 'is',
                 'not', 'or', 'the', 'these', 'this', 'to', 'with']

    # TODO: choose which functions you want to run to process the input documents
    # Again see ex2020 and ex2021 for examples
    # There are plenty of code in 'text_cleaning.py' and 'stemmers.py' for text handling
    pipeline = [
        lambda text: remove_punctuations(text, string.punctuation),
        str.lower,
        lambda text: remove_stopwords(text, stopwords),
        lambda text: stem_text(text, 'suffix'),
        # function to split text into words, could use some more 
        # advanced function in case of HTML markup or similar
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
        for i, doc_tokens in enumerate(tokenized_docs):
            # TODO: remove (+ 1) if docs start at 0!
            doc_id = i + 1

            n_token_in_doc = Counter(doc_tokens)[token]
            if n_token_in_doc > 0:


                # TODO: ADD THE TYPE OF PAYLOAD YOU WANT HERE!!!
                #       [others/none, 'word frequency', 'word position']
                #       If specified payload-type arent in the list ^ above, 
                #       you need to create your own
                payload_type = 'word frequency'
                

                # Example of frequency payload:
                if payload_type == 'word frequency':
                    payload = n_token_in_doc
                    posting = (doc_id, payload)
                    index[token].append(posting)

                # Example of position payload:
                # Cannot calculate position from doc_tokens as there are preprocessed
                # Instead need to calculate them on purely splitted and lowercase
                # tokens:
                if payload_type == 'word position':
                    original_text = docs[i]
                    position_minimal_processing = (
                            remove_nonalphanumeric(original_text.lower()))
                    positional_tokens = position_minimal_processing.split()
                    positions = find_position(positional_tokens, token)
                    if len(positions) == 0:
                        raise ValueError('There should be returned positions, since n_token_in_doc is > 0')

                    payloads = [(doc_id, pos) for pos in positions]
                    index[token].extend(payloads)

                # TODO: ONLY IF YOU NEEDED TO CREATE A NEW TYPE OF PAYLOAD:
                # if payload_type == 'new type':
                #   ... code for creating new payload-type
            
    print_pretty_index(index)


# TODO: make main run the ex2022 function.
def main():
    ex2021()


if __name__ == '__main__':
    main()