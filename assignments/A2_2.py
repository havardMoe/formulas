import abc
from collections import Counter
from collections import UserDict as DictClass
from collections import defaultdict
import math
from typing import Dict, List

CollectionType = Dict[str, Dict[str, List[str]]]


class DocumentCollection(DictClass):
    """Document dictionary class with helper functions."""

    def total_field_length(self, field: str) -> int:
        """Total number of terms in a field for all documents."""
        return sum(len(fields[field]) for fields in self.values())

    def avg_field_length(self, field: str) -> float:
        """Average number of terms in a field across all documents."""
        return self.total_field_length(field) / len(self)

    def get_field_documents(self, field: str) -> Dict[str, List[str]]:
        """Dictionary of documents for a single field."""
        return {
            doc_id: doc[field] for (doc_id, doc) in self.items() if field in doc
        }


class Scorer(abc.ABC):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = None,
        fields: List[str] = None,
    ):
        """Interface for the scorer class.

        Args:
            collection: Collection of documents. Needed to calculate document
                statistical information.
            index: Index to use for calculating scores.
            field (optional): Single field to use in scoring.. Defaults to None.
            fields (optional): List of fields to use in scoring. Defaults to
                None.

        Raises:
            ValueError: Either field or fields need to be specified.
        """
        self.collection = collection
        self.index = index

        if not (field or fields):
            raise ValueError("Either field or fields have to be defined.")

        self.field = field
        self.fields = fields

        # Score accumulator for the query that is currently being scored.
        self.scores = None

    def score_collection(self, query_terms: List[str]):
        """Scores all documents in the collection using term-at-a-time query
        processing.

        Params:
            query_term: Sequence (list) of query terms.

        Returns:
            Dict with doc_ids as keys and retrieval scores as values.
            (It may be assumed that documents that are not present in this dict
            have a retrival score of 0.)
        """
        self.scores = defaultdict(float)  # Reset scores.
        query_term_freqs = Counter(query_terms)

        for term, query_freq in query_term_freqs.items():
            self.score_term(term, query_freq)

        return self.scores

    @abc.abstractmethod
    def score_term(self, term: str, query_freq: int):
        """Scores one query term and updates the accumulated document retrieval
        scores (`self.scores`).

        Params:
            term: Query term
            query_freq: Frequency (count) of the term in the query.
        """
        raise NotImplementedError


class SimpleScorer(Scorer):
    def score_term(self, term: str, query_freq: int) -> None:
        if term not in self.index[self.field]:
            return
        for doc_id, doc_freq in self.index[self.field][term]:
            self.scores[doc_id] += (query_freq * doc_freq)


class ScorerBM25(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        b: float = 0.75,
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25, self).__init__(collection, index, field)
        self.b = b
        self.k1 = k1

    def score_term(self, term: str, query_freq: int) -> None:
        if term not in self.index[self.field]:
            return
        # total number of documents in collection
        N = len(self.collection)
        # number of documents containing the relevant term
        n_t = len(self.index[self.field][term])
        # average lenght of used field accross all documents
        avgdl = self.collection.avg_field_length(self.field)
        idf_t = math.log(N/n_t)

        for doc_id, c_td in self.index[self.field][term]:
            doc_length = len(self.collection[doc_id][self.field])
            A = idf_t * c_td * (1 + self.k1)
            B = c_td + self.k1 * (1 - self.b + self.b * doc_length / avgdl)
            self.scores[doc_id] += A / B



class ScorerLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        field: str = "body",
        smoothing_param: float = 0.1,
    ):
        super(ScorerLM, self).__init__(collection, index, field)
        self.smoothing_param = smoothing_param

    def score_term(self, term: str, query_freq: int) -> None:
        if term not in self.index[self.field]:
            return

        c_tq = query_freq
        lam = self.smoothing_param
        # term's relative frequency across the entire collection
        P_t_C = sum(c_td for _, c_td in self.index[self.field][term])\
                    / self.collection.total_field_length(self.field)

        for doc_id in self.collection:
            c_td = sum(1 for d_term in self.collection[doc_id][self.field] 
                    if d_term == term)

            len_d = len(self.collection[doc_id][self.field])
            A = (1 - lam) * (c_td / len_d) + (lam * P_t_C)
            
            score = c_tq * math.log(A)
            
            self.scores[doc_id] += score
        



class ScorerBM25F(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: List[str] = ["title", "body"],
        field_weights: List[float] = [0.2, 0.8],
        bi: List[float] = [0.75, 0.75],
        k1: float = 1.2,
    ) -> None:
        super(ScorerBM25F, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.bi = bi
        self.k1 = k1

    def score_term(self, term: str, query_freq: int) -> None:
        if term not in self.index["body"]:
            return

        # number of documents in the collection
        N = len(self.collection)
        # Number of documents having the term in the body
        n_t = len(self.index["body"][term])
        idf_t = math.log(N/n_t)

        for doc_id in self.collection:
            psuedo_c_td = 0
            for field, weight, bi in zip(self.fields, 
                                         self.field_weights, 
                                         self.bi):
                
                avd_di = self.collection.avg_field_length(field)
                di = len(self.collection[doc_id][field])  # else 0
                Bi = (1 - bi + bi * di / avd_di)
                c_td = sum(1 for d_term in self.collection[doc_id][field] 
                        if d_term == term)

                psuedo_c_td += weight * c_td / Bi
            
            score = (idf_t) * psuedo_c_td / (self.k1 + psuedo_c_td)
            self.scores[doc_id] += score


class ScorerMLM(Scorer):
    def __init__(
        self,
        collection: DocumentCollection,
        index: CollectionType,
        fields: List[str] = ["title", "body"],
        field_weights: List[float] = [0.2, 0.8],
        smoothing_param: float = 0.1,
    ):
        super(ScorerMLM, self).__init__(collection, index, fields=fields)
        self.field_weights = field_weights
        self.smoothing_param = smoothing_param

    def score_term(self, term: str, query_freq: float) -> None:
        lam = self.smoothing_param
        for doc_id in self.collection:
            P_t_theta_d = 0
            for field, weight in zip(self.fields, self.field_weights):
                # Term frequency in document - field i
                c_tdi = sum(1 for d_term in self.collection[doc_id][field] 
                        if d_term == term)
                # number of terms in field
                di = len(self.collection[doc_id][field])
                # Term's relative frequency infield i
                P_t_di = c_tdi / di
                # Collection relative term frequency in field i
                if term not in self.index[field]:
                    P_t_Ci = 0
                else:
                    P_t_Ci = sum(c_td for _, c_td in self.index[field][term])\
                        / self.collection.total_field_length(field)
                P_t_theta_di = (1 - lam) * P_t_di + lam * P_t_Ci
                P_t_theta_d += (weight) * P_t_theta_di
                
            self.scores[doc_id] += math.log(P_t_theta_d)
