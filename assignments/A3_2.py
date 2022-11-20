import abc
import time
import numpy as np
from itertools import islice
from collections import Counter
from elasticsearch import Elasticsearch
from typing import Any, Dict, List, Union

_DEFAULT_FIELD = "body"


class Entity:
    def __init__(self, doc_id: str, stats: Dict[str, Dict[str, Any]]):
        """Representation of an entity.

        Args:
          doc_id: Document id
          stats: Term vector stats from elasticsearch. Keys are field and
            values are term and field statistics.
        """
        self.doc_id = doc_id
        self._stats = stats
        self._terms = {}

    def term_stats(
        self, term: str, field: str = _DEFAULT_FIELD
    ) -> Dict[str, Any]:
        """Term statistics including term frequency and total term frequency."""
        return self._stats[field]["terms"].get(term)

    def field_stats(self, field: str = _DEFAULT_FIELD):
        """Field statistics including sum of total term frequency."""
        return self._stats[field]["field"]

    def terms(self, field: str = _DEFAULT_FIELD) -> List[str]:
        """Reconstructed document field from indexed positional information."""
        if field in self._terms:
            return self._terms[field]

        pos = {
            token["position"]: term
            for term, tinfo in self._stats[field]["terms"].items()
            for token in tinfo["tokens"]
        }
        self._terms[field] = [None] * (max(pos.keys()) + 1)
        for p, term in pos.items():
            self._terms[field][p] = term
        return self._terms[field]

    def length(self, field: str = _DEFAULT_FIELD) -> int:
        """Length of the document field."""
        return sum(
            term["term_freq"] for term in self._stats[field]["terms"].values()
        )


class ElasticsearchCollection:
    def __init__(self, index_name):
        """Interface to an Elasticsearch index.

        Args:
          index_name: Name of the index to use.
        """
        self._index_name = index_name
        self.es = Elasticsearch()

    def baseline_retrieval(
        self, query: str, k: int = 100, field: str = None
    ) -> List[str]:
        """Performs baseline retrieval on index.

        Args:
          query: A string of text, space separated terms.
          k: Number of documents to return.
          field: If specified, match only on the specified field.

        Returns:
          A list of entity IDs as strings, up to k of them, in descending
          order of scores.
        """
        res = self.es.search(
            index=self._index_name,
            q=query if not field else None,
            query={"match": {field: query}} if field else None,
            size=k,
        )
        return [x["_id"] for x in res["hits"]["hits"]]

    def get_query_terms(self, text: str) -> List[str]:
        """Analyzes text with the same pipeline that was used for indexing
        documents. It returns None in place of a term if it was removed (e.g.,
        using stopword removal).

        Args:
          text: Text to analyze.

        Returns:
          List of terms.
        """
        tokens = self.es.indices.analyze(
            index=self._index_name, body={"text": text}
        )["tokens"]
        query_terms = [None] * (
            max(tokens, key=lambda x: x["position"])["position"] + 1
        )
        for token in tokens:
            query_terms[token["position"]] = token["token"]
        return query_terms

    def get_document(self, doc_id: str) -> Entity:
        """Generates entity representation given document id."""
        tv = self.es.termvectors(
            index=self._index_name,
            id=doc_id,
            term_statistics=True,
        )["term_vectors"]

        return Entity(
            doc_id,
            stats={
                field: {
                    "terms": tv[field]["terms"],
                    "field": tv[field]["field_statistics"],
                }
                for field in tv
            },
        )

    def index(self, collection: Dict[str, Any], settings: Dict[str, Any]):
        if self.es.indices.exists(index=self._index_name):
            self.es.indices.delete(index=self._index_name)
        self.es.indices.create(index=self._index_name, mappings=settings)
        for (doc_id, doc) in collection.items():
            self.es.index(document=doc, id=doc_id, index=self._index_name)
        time.sleep(10)


class Scorer(abc.ABC):
    def __init__(
        self,
        collection: ElasticsearchCollection,
        feature_weights=[0.85, 0.1, 0.05],
        mu: float = 100,
        window: int = 3,
    ):
        """Interface for the scorer class.

        Args:
          collection: Collection of documents. Needed to calculate document
            statistical information.
          feature_weights: Weights associated with each feature function
          mu: Smoothing parameter
          window: Window for unordered feature function.
        """
        if not sum(feature_weights) == 1:
            raise ValueError("Feature weights should sum to 1.")

        self.collection = collection
        self.feature_weights = feature_weights
        self.mu = mu
        self.window = window

    def score_collection(self, query: str, k: int = 100):
        """Scores all documents in the collection using document-at-a-time query
        processing.

        Args:
          query: Sequence (list) of query terms.
          k: Number of documents to return

        Returns:
          Dict with doc_ids as keys and retrieval scores as values. (It may be
          assumed that documents that are not present in this dict have a
          retrival score of 0.)
        """
        # IDs of top k entities from the baseline retreival function
        ent_ids = self.collection.baseline_retrieval(query, k)
        documents = [self.collection.get_document(ent_id) 
                                   for ent_id in ent_ids]
        query_terms = self.collection.get_query_terms(query)

        lT, lO, lU = self.feature_weights
        return {
            doc.doc_id: (
                lT * self.unigram_matches(query_terms, doc)
                + lO * self.ordered_bigram_matches(query_terms, doc, documents)
                + lU
                * self.unordered_bigram_matches(query_terms, doc, documents)
            )
            for doc in documents
        }

    @abc.abstractmethod
    def unigram_matches(self, query_terms: List[str], doc: Entity) -> float:
        """Returns unigram matches based on smoothed entity language model.

        Args:
          query_terms: List of query terms.
          doc: Entity for which we are calculating the score.

        Returns:
          Score for unigram matches for document
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ordered_bigram_matches(
        self, query_terms: List[str], doc: Entity, documents: List[Entity]
    ) -> float:
        """Returns ordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for ordered bigram matches for document
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unordered_bigram_matches(
        self, query_terms: List[str], doc: Entity, documents: List[Entity]
    ) -> float:
        """Returns unordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for unordered bigram matches for document
        """
        raise NotImplementedError

def num_bigram_occurrences_ordered(
    bigrams: Union[List[List[str]], List[str]], 
    doc: Entity,
    sum_occurrences = False,
    field=_DEFAULT_FIELD) -> Union[List[int], int]:
    """Returns the number of times a bigram(s) occurs in an entity

    Args:
      bigram: List of Bigrams consisting of two terms OR a single Bigram
      doc: Entity where we calculate the number of occurences of the bigram
      sum: If you want to sum the number of occurences for each bigram

    Returns:
      List: Number of times each bigram is found in the entity
    """
    if len(bigrams) == 2 and isinstance(bigrams[0], str):
        bigrams = [bigrams]
    
    doc_terms = doc.terms(field=field)
    doc_bigrams_counted = Counter(
        zip(doc_terms, islice(doc_terms, 1, None))
    )

    occs = [doc_bigrams_counted[bigram] for bigram in bigrams]

    return sum(occs) if sum_occurrences else occs
   

def num_bigram_occurrences_unordered(
    bigrams: Union[List[List[str]], List[str]], 
    doc: Entity,
    window_size: int,
    sum_occurrences = False,
    field=_DEFAULT_FIELD) -> Union[List[int], int]:
    """Returns the number of times a bigram(s) occurs in an entity within a 
    window size

    Args:
    bigram: List of Bigrams consisting of two terms OR a single Bigram
    doc: Entity where we calculate the number of occurences of the bigram
    window_size: size of the window to look for bigram mathces
    sum: If you want to sum the number of occurences for each bigram

    Returns:
    List: Number of times each bigram is found in the entity
    """
    # If a single bigram is given as input
    if len(bigrams) == 2 and isinstance(bigrams[0], str):
        bigrams = [bigrams]
    
    occurrences = []
    for bigram in bigrams:
        count = 0
        doc_terms = doc.terms(field=field)
        for i in range(len(doc_terms)):
            if doc_terms[i] in bigram:
                t = bigram[0] if doc_terms[i] == bigram[1] else bigram[1]
                count += Counter(doc_terms[i+1:i+window_size])[t]
            
        occurrences.append(count)

    return sum(occurrences) if sum_occurrences else occurrences


class SDMScorer(Scorer):
    def unigram_matches(self, query_terms: List[str], doc: Entity) -> float:
        """Returns unigram matches based on smoothed entity language model.

        Args:
          query_terms: List of query terms.
          doc: Entity for which we are calculating the score.

        Returns:
          Score for unigram matches for document
        """
        # number of words in document
        le = doc.length()
        # total number of words in collection
        sum_ttf = doc.field_stats()['sum_ttf']

        total_term_freqs = []  # ttf for each term in query
        doc_term_freqs = []  # term freq in this document for each term in query
        for term in query_terms:
            # Find TTF for term
            termdoc_id = self.collection.baseline_retrieval(query=term, k=1)
            if len(termdoc_id) > 0:
                termdoc = self.collection.get_document(termdoc_id[0])
                total_term_freqs.append(termdoc.term_stats(term).get('ttf', 0))
            else:
                total_term_freqs.append(0)

            # Find document frequency of term
            doc_stats = doc.term_stats(term)
            doc_term_freq = doc_stats['term_freq'] if doc_stats else 0
            doc_term_freqs.append(doc_term_freq)
        
        # P(qi |  epsi)
        # = num occurence of term in collection / num words in colection
        P_qis = [(ttf / sum_ttf) for ttf in total_term_freqs]

        ft = [(c_qi + self.mu * P_qi) / (le + self.mu)
            for c_qi, P_qi in zip(doc_term_freqs, P_qis)]

        return sum(np.log(f) for f in ft if f > 0)

 
    def ordered_bigram_matches(
        self,
        query_terms: List[str],
        doc: Entity,
        documents: List[Entity],
    ) -> float:
        """Returns ordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for ordered bigram matches for document
        """
        query_bigrams = list(zip(query_terms, islice(query_terms, 1, None)))
        # Number of times each query_bigram occurs in the document to be scored
        c_qis_e = num_bigram_occurrences_ordered(query_bigrams, doc)

        # Number of times each query bigram occurs in the whole colelction
        c_qis_collection = []
        for bigram in query_bigrams:
            num_bigram_occs = 0
            for d in documents:
                num_bigram_occs += num_bigram_occurrences_ordered(bigram, d, 
                                                   sum_occurrences=True)
            c_qis_collection.append(num_bigram_occs)

        # total number of words in whole collection
        sum_le = doc.field_stats()['sum_ttf']
        # Lenght of document
        le = doc.length()
        # MLE for bigram in collection
        P_qis = [c_qi / sum_le for c_qi in c_qis_collection]

        fo = [(c_qi_e + self.mu * P_qi) / (le + self.mu)
              for c_qi_e, P_qi in zip(c_qis_e, P_qis)]

        return sum(np.log(f) for f in fo if f > 0)


    def unordered_bigram_matches(
        self,
        query_terms: List[str],
        doc: Entity,
        documents: List[Entity],
    ) -> float:
        """Returns unordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for unordered bigram matches for document
        """
        query_bigrams = list(zip(query_terms, islice(query_terms, 1, None)))
        # Number of times each query_bigram occurs in the document to be scored
        c_qis_e = num_bigram_occurrences_unordered(query_bigrams, doc, self.window)
        # Number of times each query bigram occurs in the whole colelction
        
        c_qis_collection = []
        for bigram in query_bigrams:
            num_bigram_occs = 0
            for d in documents:
                num_bigram_occs += (
                    num_bigram_occurrences_unordered(
                        bigram, d, self.window, True
                    )
                )
            c_qis_collection.append(num_bigram_occs)

        # total number of words in whole collection
        sum_le = doc.field_stats()["sum_ttf"]
        # Lenght of document
        le = doc.length()
        # MLE for bigram in collection
        P_qis = [c_qi / sum_le for c_qi in c_qis_collection]

        fu = [(c_qi_e + self.mu * P_qi) / (le + self.mu)
              for c_qi_e, P_qi in zip(c_qis_e, P_qis)]

        return sum(np.log(f) for f in fu if f > 0)



class FSDMScorer(Scorer):
    def __init__(
        self,
        collection: ElasticsearchCollection,
        feature_weights=[0.85, 0.1, 0.05],
        mu: float = 100,
        window: int = 3,
        fields: List[str] = ["title", "body", "anchors"],
        field_weights: List[float] = [0.2, 0.7, 0.1],
    ):
        """Fielded version of an SDM scorer.

        Args:
          collection: Collection of documents. Needed to calculate document
            statistical information.
          feature_weights: Weights associated with each feature function
          mu: Smoothing parameter
          window: Window for unordered feature function.
          fields: A list of fields to use for the calculating the score
          field_weights: A list of weights to use for each field.
        """
        super().__init__(collection, feature_weights, mu, window)
        assert len(fields) == len(field_weights)
        self.fields = fields
        self.field_weights = field_weights

    def unigram_matches(self, query_terms: List[str], doc: Entity) -> float:
        """Returns unigram matches based on smoothed entity language model.

        Args:
          query_terms: List of query terms.
          doc: Entity for which we are calculating the score.

        Returns:
          Score for unigram matches for document
        """
        # The smoothed entity language model for field
        """Returns unigram matches based on smoothed entity language model.

        Args:
          query_terms: List of query terms.
          doc: Entity for which we are calculating the score.

        Returns:
          Score for unigram matches for document
        """
        # The smoothed entity language model for field
        sum_ft = 0

        for term in query_terms:

            sum_term_ft = 0
            for f, field_w in zip(self.fields, self.field_weights):

                doc_length = doc.length(f)
                # total number of words in this field in collection
                tot_f_terms = doc.field_stats(f)["sum_ttf"]
                
                # Find TTF
                termdocs = self.collection.baseline_retrieval(query=term, 
                                                              k=1, field=f)
                ttf = 0
                if len(termdocs) > 0:
                    termdoc = self.collection.get_document(termdocs[0])
                    f_t_stats = termdoc.term_stats(term, field=f)
                    ttf = f_t_stats.get("ttf", 0) if f_t_stats else 0

                # Find document frequency of term
                doc_stats = doc.term_stats(term, field=f)
                doc_term_freq = doc_stats["term_freq"] if doc_stats else 0

                # P(qi | epsi) (field specific)
                # Num occurence of term in collection / num words in collection
                P_qi_fi = ttf / tot_f_terms
                ft_qi_fi = (doc_term_freq + self.mu * P_qi_fi) / (doc_length + self.mu)
                sum_term_ft += field_w * ft_qi_fi

            sum_ft += np.log(sum_term_ft) if sum_term_ft > 0 else 0

        return sum_ft


    def ordered_bigram_matches(
        self, query_terms: List[str], doc: Entity, documents: List[Entity]
    ) -> float:
        """Returns ordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for ordered bigram matches for document
        """
        sum_fo = 0

        query_bigrams = list(zip(query_terms, islice(query_terms, 1, None)))
        for bigram in query_bigrams:

            sum_term_fo = 0
            for f, field_w in zip(self.fields, self.field_weights):
                doc_length = doc.length(f)
                # total number of words in this field in collection
                tot_f_terms = doc.field_stats(f)["sum_ttf"]
                # Find document frequency of term
                doc_bigram_freq = num_bigram_occurrences_ordered(
                             bigram, doc, field=f, sum_occurrences=True)
                
                # Find Total bigram frequency in collection
                tbf = 0
                for d in documents:
                    tbf += num_bigram_occurrences_ordered(bigram, d, 
                                       field=f, sum_occurrences=True)

                # P(qi | epsi) (field specific)
                # Num occurence of term in collection / num words in collection
                P_qi_fi = tbf / tot_f_terms if tot_f_terms > 0 else 0
                
                fo_qi_fi = (doc_bigram_freq + self.mu * P_qi_fi)\
                                / (doc_length + self.mu)
                sum_term_fo += field_w * fo_qi_fi

            sum_fo += np.log(sum_term_fo) if sum_term_fo > 0 else 0

        return sum_fo

    def unordered_bigram_matches(
        self, query_terms: List[str], doc: Entity, documents: List[Entity]
    ) -> float:
        """Returns unordered bigram matches based on smoothed entity language
        model.

        Args:
          query_terms: List of query terms
          doc: Entity we wish to score
          documents: List of all entities in the collection

        Returns:
          Score for unordered bigram matches for document
        """
        sum_fu = 0

        query_bigrams = list(zip(query_terms, islice(query_terms, 1, None)))
        for bigram in query_bigrams:

            sum_term_fu = 0
            for f, field_w in zip(self.fields, self.field_weights):
                doc_length = doc.length(f)
                # total number of words in this field in collection
                tot_f_terms = doc.field_stats(f)["sum_ttf"]
                # Find document frequency of term
                doc_bigram_freq = num_bigram_occurrences_unordered(
                        bigram, doc, field=f, window_size=self.window, 
                        sum_occurrences=True)
                
                # Find Total bigram frequency in collection
                tbf = 0
                for d in documents:
                    tbf += num_bigram_occurrences_unordered(bigram, d, 
                            field=f, window_size=self.window, 
                            sum_occurrences=True)

                # P(qi | epsi) (field specific)
                # Num occurence of term in collection / num words in collection
                P_qi_fi = tbf / tot_f_terms if tot_f_terms > 0 else 0
                
                fu_qi_fi = (doc_bigram_freq + self.mu * P_qi_fi)\
                                / (doc_length + self.mu)
                sum_term_fu += field_w * fu_qi_fi

            sum_fu += np.log(sum_term_fu) if sum_term_fu > 0 else 0

        return sum_fu
