from typing import List
import numpy as np


class NaiveBayesClassifier:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        # holds P(class)
        self.class_priors = [0.] * (num_classes)
        # holds P(term | class) (using Laplace (add-one) smoothing)
        self.term_probs: List[List[float]]
        print('''
+------------------------------------------------------+
|NB NB NB NB NB NB NB NB NB NB NB NB NB NB NB NB NB NB |
|                                                      |
|DO USE TERM NUMBERS AND CLASS-LABELS THAT START WITH  |
|0 -0 -0 -0 -0 -0 -0 -0 -0 -0 -0 -0 -0 -0 -0 -0 -0 - 0 |
|                                                      |
|NB NB NB NB NB NB NB NB NB NB NB NB NB NB NB NB NB NB |
+------------------------------------------------------+''')

    def fit(self, doc_terms: List[List[int]], labels: List[int]):
        vocab_length = len(doc_terms[0])

        # Calculating P(class) class priors
        for l in set(labels):
            self.class_priors[l] = labels.count(l) / len(labels)

        # Calculating P(term | class)
        self.term_probs = np.zeros((self.num_classes, vocab_length))
        for term_index in range(vocab_length):
            for c in range(self.num_classes):
                # Calculate P(term | class)
                count_inclass = self._termcount_in_class(doc_terms, labels, 
                                                         term_index, c)
                count_collection = self._total_terms_inclass(
                        doc_terms, labels, c)
                self.term_probs[c][term_index] = (
                    (count_inclass + 1) / (count_collection + self.num_classes)
                )

    def classify_doc(self, doc: List[int]):
        probs = [self.p_class_doc(doc, c_i) for c_i in range(self.num_classes)]
        print('probs: ', probs)
        print('-' * 23, 'warning:', '-' * 23)
        print('NB Classes start at 0!!!')
        print('Remember to add (+1) to prediction if document-ids are starting at 1')
        print('-' * 56)
        return np.argmax(probs)


    def p_class_doc(self, doc: List[int], c: int):
        p = 1
        for term_index in range(len(doc)):
            doc_occ = doc[term_index]
            if doc_occ > 0:
                p *= (self.p_term_class(term_index, c) * doc_occ)

        return self.p_class_prior(c) * p

    def p_class_prior(self, c: int):
        return self.class_priors[c]

    def p_term_class(self, term_index: int, c: int):
        return self.term_probs[c][term_index]
    
    def _termcount_in_collection(self, doc_terms: List[List[int]], 
                                 term_index: int):
        tc = 0
        for terms in doc_terms:
            tc += terms[term_index]
        return tc
    
    def _termcount_in_class(self, doc_terms: List[List[int]], 
                            labels: List[int], term_index: int, c: int):
        tc = 0
        for terms, label in zip(doc_terms, labels):
            if label == c:
                tc += terms[term_index]

        return tc
    
    def _total_terms_inclass(self, doc_terms: List[List[int]], 
                             labels: List[int], c: int):
        count = 0
        for terms, label in zip(doc_terms, labels):
            if label != c:
                continue
            count += sum(terms)
        return count

    def _total_terms_in_collection(self, doc_terms: List[List[int]]):
        N = 0
        for terms in doc_terms:
            N += sum(terms)
        return N

def ex2020():
    print()
    print('-------- Exam 2020 results --------')

    terms = [[1, 9, 7, 5, 2, 1, 3, 0],
             [0, 2, 1, 3, 0, 1, 0, 2],
             [1, 1, 0, 0, 4, 5, 2, 1],
             [2, 1, 4, 0, 5, 2, 7, 0],
             [0, 1, 5, 0, 1, 4, 0, 2],
             [6, 3, 0, 3, 2, 1, 3, 0],
             [2, 1, 3, 1, 8, 1, 0, 9],
             [1, 1, 0, 1, 0, 9, 0, 0]]

    # TODO: REMEMBER TO USE LABELS THAT START WITH 0
    labels = [l-1 for l in [1, 2, 1, 3, 2, 3, 1, 3]]
    num_classes = max(labels) + 1

    nbl = NaiveBayesClassifier(num_classes)
    nbl.fit(terms, labels)
    
    print('#1 P(C2)')
    print(nbl.p_class_prior(1))
    print()

    print('#2 P(t6 | C2)')
    print(nbl.p_term_class(5, 1))
    print()

    print('#3 P(C2| new_doc)')
    print('new_doc = "t2 t3 t6"')
    new_doc = [0, 1, 1, 0, 0, 1, 0, 0]
    print(nbl.p_class_doc(new_doc, c=1))
    print()

    print('#4 classify(new_doc)')
    print('new_doc = "t2 t7 t8"')
    new_doc = [0, 1, 0, 0, 0, 0, 1, 1]
    print(nbl.classify_doc(new_doc))


def ex2021():
    print('------- Exam 2021 results ---------')
    terms = [[2, 0, 1, 2, 0, 2, 4],
             [0, 0, 0, 0, 3, 2, 2],
             [3, 4, 0, 2, 0, 0, 2],
             [4, 0, 3, 1, 1, 1, 0],
             [1, 0, 0, 3, 1, 2, 0],
             [0, 1, 1, 0, 3, 4, 1]]

    # TODO: REMEMBER TO USE LABELS THAT START WITH 0
    labels = [l-1 for l in [1, 3, 2, 3, 2, 1]]
    num_classes = max(labels) + 1

    nbl = NaiveBayesClassifier(num_classes)
    nbl.fit(terms, labels)
    
    print('# 1 P(C2)')
    print('class prior')
    print(nbl.p_class_prior(1))
    print()

    print('# 2 P(t4 | C2)')
    print(nbl.p_term_class(3, 1))
    print()

    print('# 3 P(C1 | new_doc)')
    print('new_doc = "t1"')
    new_doc = [1, 0, 0, 0, 0, 0, 0]
    print(nbl.p_class_doc(new_doc, c=0))
    print()

    print('#4 - P(C3 | new_doc)')
    print('new_doc = "t1 t4 t5"')
    new_doc = [1, 0, 0, 1, 1, 0, 0]
    print(nbl.p_class_doc(new_doc, c=2))
    print()

    print('#5 - classify(new_doc)')
    print('new_doc = "t4 t4 t5"')
    new_doc = [0, 0, 0, 2, 1, 0, 0]
    print(f'predicted class: {nbl.classify_doc(new_doc)}')

def ex2022():
    # See ex2021 and 2020 for detaion on how to use
    # INSERT INFO FROM 2022 exam here!
    print('------- Exam 2022 results ---------')

    # TODO: insert terms
    # num_docs x len_vocab matrix
    terms = [[ ... ],
             [ ... ],
             [ ... ],
             [ ... ],
             [ ... ],
             [ ... ]]

    # REMEMBER TO USE LABELS THAT START WITH 0
    # TODO: insert labels in [ ... ] list below
    labels = [l-1 for l in [ ... ]]
    num_classes = max(labels) + 1

    nbl = NaiveBayesClassifier(num_classes)
    nbl.fit(terms, labels)
    
    print('#1 -  ... ')
    # TODO: code to calculate task#1 here
    print()

    print('#2 -  ... ')
    # TODO: code to calculate task#2 here
    print()

    print('#3 -  ...')
    # TODO: code to calculate task#3 here
    print()

    print('#4 - ...')
    # TODO: code to calculate task#4 here
    print()

    print('#5 - ...')
    # TODO: code to calculate task#5 here

# Note: ex2021 is just here as an example
# TODO: run ex2022 function inside main
def main():
    ex2021()

    
if __name__ == '__main__':
    main()