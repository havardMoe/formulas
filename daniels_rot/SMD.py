import abc
import time
from typing import Any, Dict, List
import math
from elasticsearch import Elasticsearch
from collections import Counter


DOCS = {1: {"body": "t1 t2 t3 t4 t5 t6"},
        2: {"body": "t3 t4 t3 t1 t8 t2 t2 t7"},
        3: {"body": "t2 t9 t4 t1 t8 t2 t3 t1 t4"}}

q = ['t4 t3']

INDEX_NAME = "toy_index_1"

INDEX_SETTINGS = {
    "properties": {
        "title": {
            "type": "text",
            "term_vector": "with_positions",
            "analyzer": "english",
        },
        "body": {
            "type": "text",
            "term_vector": "with_positions",
            "analyzer": "english",
        },
        "anchors": {
            "type": "text",
            "term_vector": "with_positions",
            "analyzer": "english",
        },
    }
}

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
        self.es = Elasticsearch(hosts=['http://localhost:9200/'])

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
        mu: float = 6,
        window: int = 4,
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

        res = self.collection.es.search(index=self.collection._index_name, doc_type="_doc", body = {'query': {'match_all' : {}}})['hits']['hits']
        docs = [doc['_id'] for doc in res]

        documents: List[Entity] = [self.collection.get_document(id) for id in docs]
        query_terms: List[str] = query.split()

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


class SDMScorer(Scorer):
    def unigram_matches(self, query_terms: List[str], doc: Entity) -> float:
      """Returns unigram matches based on smoothed entity language model.
      Args:
        query_terms: List of query terms.
        doc: Entity for which we are calculating the score.
      Returns:
        Score for unigram matches for document
      """
      score = 0
      length = doc.length()
      total_length = doc.field_stats()['sum_ttf']

      for term in query_terms:
        total_term_freq_stat = self.collection.baseline_retrieval(query=term,k=1)
        term_freq = doc.term_stats(term)['term_freq'] if doc.term_stats(term) != None else 0
        total_term_freq = self.collection.get_document(total_term_freq_stat).term_stats(term)['ttf'] if len(total_term_freq_stat)>0 else 0
        p = total_term_freq/total_length if total_length >0 else 0
        tmp = (term_freq+self.mu*(p))/(length + self.mu) 
        score += math.log(tmp,2) if tmp > 0 else 0
      return score

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
      score = 0
      total_freq = 0
      terms = doc.terms()
      total_length = doc.field_stats()['sum_ttf']
      length = doc.length()
      bigram = [x for x in zip(query_terms[:-1],query_terms[1:])]
      term_freq = sum([1 for i in bigram for c in range(len(terms)-1) if terms[c]==i[0]and terms[c+1]==i[1]])
  
      for term in bigram:
        docs =[self.collection.get_document(d) for d in self.collection.baseline_retrieval(query=term)]
        for x in docs:
          total_freq += sum([1 for c in range(len(x.terms())-1) if x.terms()[c]==term[0]and x.terms()[c+1]==term[1]])

      p = total_freq/total_length if total_length >0 else 0
      tmp = (term_freq+self.mu*(p))/(length + self.mu)
      score += math.log(tmp,2) if tmp >0 else 0
      return score


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
        def freq(bigram,doc,window = self.window):
          terms = doc.terms()
          count = 0
          for i in range(len(terms)-1):
            if terms[i] in bigram:
              other_term = bigram[0] if terms[i] == bigram[1] else bigram[1]
              count += Counter(terms[i+1:i+window])[other_term]
          return count

        bigrams = [x for x in zip(query_terms[:-1],query_terms[1:])]
        total_length = doc.field_stats()['sum_ttf']
        length = doc.length ()
        score = 0

        for gram in bigrams:
          docs =[self.collection.get_document(d) for d in self.collection.baseline_retrieval(query=gram)]
          term_freq = freq(gram,doc)
          total_term_freq = sum([freq(gram,doc_in_col) for doc_in_col in docs])
          p = total_term_freq/total_length if total_length >0 else 0
          tmp = (term_freq+self.mu*p)/(length+self.mu)
          score += math.log(tmp,2) if tmp > 0 else 0
        return score

    def total_doc_score(self,query,document):
        
        uni = sum([self.unigram_matches([term], document)for term in query])
        order = self.ordered_bigram_matches(query_terms=query,doc=document,documents=self.collection)
        unordered = self.unordered_bigram_matches(query_terms=query,doc=document,documents=self.collection)
        return (self.feature_weights[0]*uni)+(self.feature_weights[1]*order)+(self.feature_weights[2]*unordered)



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
        score = 0

        for term in query_terms:
          field_score = 0
          for field, weight in zip(self.fields,self.field_weights):
            length = doc.length(field)
            total_field_terms = doc.field_stats(field)["sum_ttf"]

            total_term_freq_stat = self.collection.baseline_retrieval(query=term,k=1,field=field)
            total_term_freq = self.collection.get_document(total_term_freq_stat).term_stats(term,field=field)['ttf'] if len(total_term_freq_stat)>0 else 0
            
            doc_stat = doc.term_stats(term,field=field)
            doc_term_freq = doc_stat["term_freq"] if doc_stat else 0

            p = total_term_freq / total_field_terms
            field_score += weight*((doc_term_freq + self.mu*p) / (length + self.mu))
            
          score +=math.log(field_score) if field_score >0 else 0 
        return score


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
        score = 0
        bigram = [x for x in zip(query_terms[:-1],query_terms[1:])]
        for gram in bigram:
          field_score = 0
          for field, weigth in zip(self.fields,self.field_weights):
            total_freq = 0
            terms = doc.terms(field=field)
            field_freq = sum([1 for i in range(len(terms)-1) if terms[i]==gram[0]and terms[i+1]==gram[1]])
            total_field_length = doc.field_stats(field=field)['sum_ttf']
            docs =[self.collection.get_document(d) for d in self.collection.baseline_retrieval(query=gram)]
            
            for x in docs:
              total_freq += sum([1 for c in range(len(x.terms(field=field))-1) if x.terms(field=field)[c]==gram[0]and x.terms(field=field)[c+1]==gram[1]])
            
            p = total_freq/total_field_length
            field_score += weigth *((field_freq+self.mu*p)/(len(terms)+self.mu))

          score+=math.log(field_score) if field_score > 0 else 0
        return score
      
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

        def freq(bigram,doc,window = self.window, field = _DEFAULT_FIELD):
          terms = doc.terms(field = field)
          count = 0
          for i in range(len(terms)-1):
            if terms[i] in bigram:
              other_term = bigram[0] if terms[i] == bigram[1] else bigram[1]
              count += Counter(terms[i+1:i+window])[other_term]
          return count
        score = 0
        
        bigram = [x for x in zip(query_terms[:-1],query_terms[1:])]
        for gram in bigram:
          field_score = 0
          for field, weigth in zip(self.fields,self.field_weights):
            term_freq = freq(gram,doc,field=field)
            docs =[self.collection.get_document(d) for d in self.collection.baseline_retrieval(query=gram)]
            total_term_freq = sum([freq(gram,doc_in_col,field=field) for doc_in_col in docs])
            total_field_length = doc.field_stats(field=field)['sum_ttf']

            p = total_term_freq/total_field_length
            field_score += weigth *((term_freq+self.mu*p)/(len(doc.terms(field = field))+self.mu))

          score+=math.log(field_score) if field_score > 0 else 0
        return(score)



if __name__ == '__main__':
    mu = 6
    window = 4
    weights = [0.85,0.1,0.05]

    query = ['t4','t3']


    collection = ElasticsearchCollection("toy_index_1")
    collection.index(DOCS, INDEX_SETTINGS)
    documents = [collection.get_document(doc_id) for doc_id in DOCS.keys()]
    SDM = SDMScorer(mu=mu,window=window,feature_weights=weights,collection=collection)
   
    #print(d[2].doc_id)
    print(SDM.unigram_matches(["t4"], documents[2]))
    print(SDM.ordered_bigram_matches(query_terms=["t4","t3"],doc=documents[2],documents=documents))
    print(SDM.unordered_bigram_matches(query_terms=["t4","t3"],doc=documents[2],documents=documents))

    doc_1 = SDM.total_doc_score(query=query,document=documents[0])
    doc_2 = SDM.total_doc_score(query=query,document=documents[1])
    doc_3 = SDM.total_doc_score(query=query,document=documents[2])

    print(f'Doc 1: {doc_1}\nDoc 2: {doc_2}\nDoc 3: {doc_3}')


