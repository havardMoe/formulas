import os
import re
from typing import Any, List, Tuple, Dict
from collections import namedtuple, defaultdict, Counter

import ir_datasets
import nltk
import requests
from sqlitedict import SqliteDict

nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

WAPO_FIELDS: Dict[str, int] = {
    "doc_id": 0,
    "url": 1,
    "title": 2,
    "author": 3,
    "published_date": 4,
    "kicker": 5,
    "body": 6,
    "body_paras_html": 7,
    "body_media": 8
}


def download_dataset(filename: str, force: bool = False) -> None:
    """Download a dataset to be used with ir_datasets.

    Args:
        filename: Name of the file to download.
        force (optional): Downloads a file and overwrites if already exists.
            Defaults to False.
    """
    filepath = os.path.expanduser(f"~/.ir_datasets/wapo/{filename}")
    if force or not os.path.isfile(filepath):
        print("File downloading...")
        response = requests.get(f"https://gustav1.ux.uis.no/dat640/{filename}")
        if response.ok:
            print("File downloaded; saving to file...")
        with open(filepath, "wb") as f:
            f.write(response.content)

    print("First document:\n")
    print(next(ir_datasets.load("wapo/v2/trec-core-2018").docs_iter()))


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
        return self.index[field][term]

    def get_term_frequency(self, field: str, term: str, doc_id: str) -> int:
        """Return the frequency of a given term in a document.

        Args:
            field: Index field.
            term: Term for which to find the count.
            doc_id: Document ID

        Returns:
            Term count in a document.
        """
        try:
            return next(count for (doc_id, count) 
                        in self.index[field][term] 
                        if doc_id==doc_id)
        except StopIteration:
            return 0

    def get_terms(self, field: str) -> List[str]:
        """Returns all unique terms in the index.

        Args:
            field: Field for which to return the terms.

        Returns:
            Set of all terms in a given field.
        """
        if not field in self.index.keys():
            raise ValueError("field doesn't exist in index")
        return self.index[field].keys()

    def __exit__(self, *exc_info):
        if self.flag == "n":
            self.update(self.index)
            self.commit()
            print("Index updated.")
        super().__exit__(*exc_info)

def index_collection(
    collection: str = "wapo/v2/trec-core-2018",
    filename: str = "inverted_index.sqlite",
    num_documents: int = 595037,
) -> None:
    """Builds an inverted index from a document collection.

    Note: WashingtonPost collection has 595037 documents. This might take a very
        long time to index on an average computer.


    Args:
        collection: Collection from ir_datasets.
        filename: Sqlite filename to save index to.
        num_documents: Number of documents to index.
    """
    index_fields = ["title", "body"]
    dataset = ir_datasets.load(collection)
    with InvertedIndex(filename, new=True) as index:
        # Adding the relevant fields (i.e. title and body)
        # as its own dicts inside the index, each of these dicts will be of type: 
        # 'term': [..., (docid_i, count_i), (docid_i+1, count_i+1), ...]
        for field in index_fields:
            index.index[field] = defaultdict(list)

        for i, doc in enumerate(dataset.docs_iter()):
            if (i + 1) % (num_documents // 100) == 0:
                print(f"{round(100*(i/num_documents))}% indexed.")
            if i == num_documents:
                break
            doc_id = doc.doc_id
            for field in index_fields:
                words = preprocess(doc[WAPO_FIELDS[field]])
                for term, freq in Counter(words).items():
                    index.index[field][term].append((doc_id, freq))


if __name__ == "__main__":
    download_dataset("WashingtonPost.v2.tar.gz")
    collection = "wapo/v2/trec-core-2018"
    index_file = "inverted_index.sqlite"

    # Comment out this line or delete the index file to re-index.
    if not os.path.isfile(index_file):
        # There are total 595037 documents in the collection.
        # Consider using a smaller subset while developing the index because
        # indexing the entire collection might take a considerable amount of
        # time.
        index_collection(collection, index_file, 1000)

    index = InvertedIndex(index_file)
    print(len(index.get_postings("body", "norway")))
    print(
        index.get_term_frequency("body", "norway", "ebff82c9cd96407d2ef1ba620313f011")
    )
    index.close()
