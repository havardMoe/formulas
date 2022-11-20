import os
import re
from typing import Any, List
from sqlitedict import SqliteDict
from more_itertools import locate

DOCS = {
    1: {"title": "",
        "body": 'If you\'re going to San Francisco be sure to wear some flowers in your hair'
        },
    2: {"title": "",
        "body": 'My Loard Bassanio, since you have found Antonio, we two will leave you: but at dinner-time, I pray you, have in mind where we must meet.'
        },
    3: {"title": "",
        "body": 'A federal judge\'s approval of T-Mobile\'s takeover of Sprint will test whether three giants will compete as aggressively for cellphone users as four unequal players once did.'
        },
} 

STOPWORDS = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]


def original_to_list(text):
    chars = ["'", ".", ":", ",", "!","-", "?", "(", ")"]
    for ch in chars:
        text = text.replace(ch, " ").lower()
    text = text.split()
    return text

def find_indices(list_to_check, item_to_find):
    indices = list(locate(list_to_check, lambda x: x == item_to_find))
    indices = [ind+1 for ind in indices]
    return indices


def preprocess(doc: str) -> List[str]:
    """Preprocesses a string of text.

    Arguments:
        doc: A string of text.

    Returns:
        List of strings.
    """
    return [
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
        if term not in STOPWORDS
    ] if doc is not None else [] 


class InvertedIndex(SqliteDict):
    def __init__(
        self,
        filename: str = "inverted_index.sqlite",
        fields: List[str] = ["title", "body"],
        new: bool = False,
    ) -> None:
        super().__init__(filename, flag="n" if new else "c")
        self.fields = fields
        self.index = {} if new else self

    def get_postings(self, field: str, term: str) -> List[Any]:
        """Fetches the posting list for a given field and term.

        Args:
            field: Field for which to get postings.
            term: Term for which to get postings.

        Returns:
            List of postings for the given term in the given field.
        """
        return [(k,v) for k,v in zip(self.index[field][term].keys(),self.index[field][term].values())]

    def get_term_frequency(self, field: str, term: str, doc_id: str) -> int:
        """Return theebff82c9cd96407d2ef1ba620313f011
            field: Index field.
            term: Term for which to find the count.
            doc_id: Document ID

        Returns:
            Term count in a document.
        """
        return self.index[field][term][doc_id]

    def get_terms(self, field: str) -> List[str]:
        """Returns all unique terms in the index.

        Args:
            field: Field for which to return the terms.

        Returns:
            Set of all terms in a given field.
        """
        return self.index[field].keys()

    def __exit__(self, *exc_info):
        if self.flag == "n":
            self.update(self.index)
            self.commit()
            print("Index updated.")
        super().__exit__(*exc_info)


def index_collection(
    collection = DOCS,
    filename: str = "inverted_index.sqlite",
) -> None:
    """Builds an inverted index from a document collection.

    Note: WashingtonPost collection has 595037 documents. This might take a very
        long time to index on an average computer.


    Args:
        collection: Collection from ir_datasets.
        filename: Sqlite filename to save index to.
        num_documents: Number of documents to index.
    """

    with InvertedIndex(filename, new=True) as index:
        index.index['title'] = {}
        index.index['body'] = {}
        for id, doc in collection.items():
            
            title = preprocess(doc['title'])
            body = preprocess(doc['body'])

            original_title = original_to_list(doc['title'])
            original_body = original_to_list(doc['body'])


            for word in title:
                if word in index.index['title']:
                    index.index['title'][word].update({id : find_indices(original_title,word)})#original_title.index(word)+1})
                else:
                    index.index['title'][word] = {id: find_indices(original_title,word)}

            for w in body:
                if w in index.index['body']:
                    index.index['body'][w].update({id: find_indices(original_body,w)})
                else:
                    index.index['body'][w] = {id: find_indices(original_body,w)}


if __name__ == "__main__":
    index_file = "inverted_index.sqlite"

    # Comment out this line or delete the index file to re-index.
    if not os.path.isfile(index_file):
        # There are total 595037 documents in the collection.
        # Consider using a smaller subset while developing the index because
        # indexing the entire collection might take a considerable amount of
        # time.
        index_collection(DOCS, index_file)

    index = InvertedIndex(index_file)

    index.close()

    db = SqliteDict("inverted_index.sqlite")
    print("There are %d items in the database" % len(db))

    for key, item in db.items():
        print("%s=%s" % (key, item))

    db.close()
