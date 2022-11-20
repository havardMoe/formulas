import json
import numpy as np

from typing import Any, Callable, Dict, List, Set, Tuple, Union
from collections import defaultdict
from elasticsearch import Elasticsearch
from html.parser import HTMLParser
from sklearn.linear_model import LinearRegression

INDEX_NAME = 'trec9_index'
DATA_PATH = 'data/{}'

FIELDS = ['title', 'body']

INDEX_SETTINGS = {
    'properties': {
        'title': {'type': 'text', 'term_vector': 'yes', 'analyzer': 'english'},
        'body': {'type': 'text', 'term_vector': 'yes', 'analyzer': 'english'},
    }
}

FEATURES_QUERY = [
    'query_length',
    'query_sum_idf',
    'query_max_idf',
    'query_avg_idf',
]
FEATURES_DOC = ['doc_length_title', 'doc_length_body']
FEATURES_QUERY_DOC = [
    'unique_query_terms_in_title',
    'sum_TF_title',
    'max_TF_title',
    'avg_TF_title',
    'unique_query_terms_in_body',
    'sum_TF_body',
    'max_TF_body',
    'avg_TF_body',
]


def analyze_query(
    es: Elasticsearch, query: str, field: str, index: str = 'toy_index'
) -> List[str]:
    '''Analyzes a query with respect to the relevant index.

    Args:
        es: Elasticsearch object instance.
        query: String of query terms.
        field: The field with respect to which the query is analyzed.
        index: Name of the index with respect to which the query is analyzed.

    Returns:
        A list of query terms that exist in the specified field among the
        documents in the index.
    '''
    tokens = es.indices.analyze(index=index, body={'text': query})['tokens']
    query_terms = []
    for t in sorted(tokens, key=lambda x: x['position']):
        # Use a boolean query to find at least one document that contains the
        # term.
        hits = (
            es.search(
                index=index,
                query={'match': {field: t['token']}},
                _source=False,
                size=1,
            )
            .get('hits', {})
            .get('hits', {})
        )
        doc_id = hits[0]['_id'] if len(hits) > 0 else None
        if doc_id is None:
            continue
        query_terms.append(t['token'])
    return query_terms


def get_doc_term_freqs(
    es: Elasticsearch, doc_id: str, field: str, index: str = 'toy_index'
) -> Dict[str, int]:
    '''Gets the term frequencies of a field of an indexed document.

    Args:
        es: Elasticsearch object instance.
        doc_id: Document identifier with which the document is indexed.
        field: Field of document to consider for term frequencies.
        index: Name of the index where document is indexed.

    Returns:
        Dictionary of terms and their respective term frequencies in the field
        and document.
    '''
    tv = es.termvectors(
        index=index, id=doc_id, fields=field, term_statistics=True
    )
    if tv['_id'] != doc_id:
        return None
    if field not in tv['term_vectors']:
        return None
    term_freqs = {}
    for term, term_stat in tv['term_vectors'][field]['terms'].items():
        term_freqs[term] = term_stat['term_freq']
    return term_freqs

# Helperfunciton    
def get_termvector(es: Elasticsearch, term: str, index: str, 
    field: str='body', none_ret: Any = {}) -> Union[dict, Any]:
    '''
    Returns termvector for given term, index and field.
    If there are no documents with the given search criteria, 
    none_ret will be returned.
    '''
    doc_with_term = es.search(
                          query={'match': {field: term}},
                          index=index, 
                          size=1
    )['hits']['hits']
    if len(doc_with_term) == 0:
        return none_ret
    
    doc_id = doc_with_term[0]['_id']
    term_vector = es.termvectors(
        index=index, id=doc_id, fields=field, term_statistics=True
    ).get('term_vectors', {})\
     .get(field, {})\
     .get('terms', {})\
     .get(term, None)
    
    return term_vector if term_vector is not None else none_ret


def extract_query_features(
    query_terms: List[str], es: Elasticsearch, index: str = 'toy_index'
) -> Dict[str, float]:
    '''Extracts features of a query.

    Args:
        query_terms: List of analyzed query terms.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.
    Returns:
        Dictionary with keys 'query_length', 'query_sum_idf',
            'query_max_idf', and 'query_avg_idf'.
    '''
    features = {
        'query_length':  0,
        'query_sum_idf': 0,
        'query_max_idf': 0,
        'query_avg_idf': 0
    }
    if len(query_terms) == 0:
        return features
    # Finding idf for each term
    num_documents = es.count(index=index).get('count', 0)
    idfs_t = []
    for term in query_terms:
        tv = get_termvector(es, term, index, field='body', none_ret={})
        doc_freq = tv.get('doc_freq', 0)
        idf_t = np.log((num_documents / doc_freq), ) if doc_freq > 0 else 0
        idfs_t.append(idf_t)

    features['query_length']  = len(query_terms)
    features['query_sum_idf'] = sum(idfs_t)
    features['query_max_idf'] = max(idfs_t)
    features['query_avg_idf'] = sum(idfs_t) / len(idfs_t)

    return features


def extract_doc_features(
    doc_id: str, es: Elasticsearch, index: str = 'toy_index'
) -> Dict[str, float]:
    '''Extracts features of a document.

    Args:
        doc_id: Document identifier of indexed document.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.

    Returns:
        Dictionary with keys 'doc_length_title', 'doc_length_body'.
    '''
    tt = es.termvectors(
        index=index, id=doc_id, term_statistics=True, fields=['body', 'title']
    )['term_vectors']

    tv = tt.get('title', {}).get('terms', {})
    bv = tt.get('body', {}).get('terms', {})

    return {
        'doc_length_title': sum(t['term_freq'] for t in tv.values()),
        'doc_length_body': sum(t['term_freq'] for t in bv.values())
    }


def extract_query_doc_features(
    query_terms: List[str],
    doc_id: str,
    es: Elasticsearch,
    index: str = 'toy_index',
) -> Dict[str, float]:
    '''Extracts features of a query and document pair.

    Args:
        query_terms: List of analyzed query terms.
        doc_id: Document identifier of indexed document.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.

    Returns:
        Dictionary with keys 'unique_query_terms_in_title',
            'unique_query_terms_in_body', 'sum_TF_title', 'sum_TF_body',
            'max_TF_title', 'max_TF_body', 'avg_TF_title', 'avg_TF_body'.
    '''
    features = {
        'unique_query_terms_in_title': 0,
        'unique_query_terms_in_body':  0,
        'sum_TF_title':                0,
        'sum_TF_body':                 0,
        'max_TF_title':                0,
        'max_TF_body':                 0,
        'avg_TF_title':                0,
        'avg_TF_body':                 0,
    }
    if len(query_terms) == 0:
        return features

    tv = es.termvectors(
        index=index, id=doc_id, fields=['body', 'title'], term_statistics=True
    )['term_vectors']

    title_terms = tv.get('title', {}).get('terms', {})

    body_terms = tv.get('body', {}).get('terms', {})

    features['unique_query_terms_in_title'] = (
        sum(1 for t in query_terms if t in title_terms)    
    )
    features['unique_query_terms_in_body'] = (
        sum(1 for t in query_terms if t in body_terms)    
    )
    features['sum_TF_title'] = (
        sum(title_terms.get(t, {}).get('term_freq', 0) for t in query_terms)
    )
    features['sum_TF_body'] = (
        sum(body_terms.get(t, {}).get('term_freq', 0) for t in query_terms)
    )
    features['max_TF_title'] = (
        max(title_terms.get(t, {}).get('term_freq', 0) for t in query_terms)
    )
    features['max_TF_body'] = (
        max(body_terms.get(t, {}).get('term_freq', 0) for t in query_terms)
    )
    features['avg_TF_title'] = features['sum_TF_title'] / len(query_terms)
    features['avg_TF_body'] = features['sum_TF_body'] / len(query_terms)


    return features


def extract_features(
    query_terms: List[str],
    doc_id: str,
    es: Elasticsearch,
    index: str = 'toy_index',
) -> List[float]:
    '''Extracts query features, document features and query-document features
    of a query and document pair.

    Args:
        query_terms: List of analyzed query terms.
        doc_id: Document identifier of indexed document.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.

    Returns:
        List of extracted feature values in a fixed order.
    '''
    query_features = extract_query_features(query_terms, es, index=index)
    feature_vect = [query_features[f] for f in FEATURES_QUERY]

    doc_features = extract_doc_features(doc_id, es, index=index)
    feature_vect.extend([doc_features[f] for f in FEATURES_DOC])

    query_doc_features = extract_query_doc_features(
        query_terms, doc_id, es, index=index
    )
    feature_vect.extend([query_doc_features[f] for f in FEATURES_QUERY_DOC])

    return feature_vect


def index_documents(filepath: str, es: Elasticsearch, index: str) -> None:
    '''Indexes documents from JSONL file.'''
    bulk_data = []
    with open(filepath, 'r') as docs:
        for doc in docs:
            doc = json.loads(doc)
            bulk_data.append(
                {'index': {'_index': index, '_id': doc.pop('doc_id')}}
            )
            bulk_data.append(doc)
    es.bulk(index=index, body=bulk_data, refresh=True)

# Helper for reading queries
class QueryParser(HTMLParser):
    def __init__(self, queries: list, tags=['num', 'title']):
        self.tags = tags
        self.current_tag = ''
        self.current_query = {}
        self.queries = queries
        super().__init__()
    
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

    def handle_endtag(self, tag):
        if tag == 'top':
            self.queries.append(self.current_query)
            self.current_tag = None
            self.current_query = {}

    def handle_data(self, data):
        if data.isspace():
            return
        if self.current_tag in self.tags:
            self.current_query[self.current_tag] = data.strip()

def clean_number(data):
    return data.replace('Number: ', '').strip()


def load_queries(filepath: str) -> Dict[str, str]:
    '''Given a filepath, returns a dictionary with query IDs and corresponding
    query strings.

    This is an example query:

    ```
    <top>
    <num> Number: OHSU1
    <title> 60 year old menopausal woman without hormone replacement therapy
    <desc> Description:
    Are there adverse effects on lipids when progesterone is given with estrogen replacement therapy
    </top>

    ```

    Take as query ID the value (on the same line) after `<num> Number: `,
    and take as the query string the rest of the line after `<title> `. Omit
    newline characters.

    Args:
        filepath: String (constructed using os.path) of the filepath to a
        file with queries.

    Returns:
        A dictionary with query IDs and corresponding query strings.
    '''
    queries = []
    parser = QueryParser(queries, tags=['num', 'title'])
    with open(filepath, 'r') as f:
        parser.feed(f.read())
    
    queries = {clean_number(q['num']): q['title'] for q in queries}

    return queries


def load_qrels(filepath: str) -> Dict[str, List[str]]:
    '''Loads query relevance judgments from a file.
    The qrels file has content with tab-separated values such as the following:

    ```
    MSH1	87056458
    MSH1	87056800
    MSH1	87058606
    MSH2	87049102
    MSH2	87056792
    ```

    Args:
        filepath: String (constructed using os.path) of the filepath to a
            file with queries.

    Returns:
        A dictionary with query IDs and a corresponding list of document IDs
            for documents judged relevant to the query.
    '''
    qrels = defaultdict(list)

    with open(filepath, 'r') as f:
        for row in f.readlines():
            split_row = row.split()

            if len(split_row) == 2:
                query_id, doc_id = split_row
                qrels[query_id].append(doc_id)

    return qrels

# Helperfunciton
def unique_ids(l1, l2):
    l = [*l1, *l2]
    seen = set()
    seen_add = seen.add
    return [x for x in l if not (x in seen or seen_add(x))]

def prepare_ltr_training_data(
    query_ids: List[str],
    all_queries: Dict[str, str],
    all_qrels: Dict[str, List[str]],
    es: Elasticsearch,
    index: str,
) -> Tuple[List[List[float]], List[int]]:
    '''Prepares feature vectors and labels for query and document pairs found
    in the training data.

        Args:
            query_ids: List of query IDs.
            all_queries: Dictionary containing all queries.
            all_qrels: Dictionary with keys as query ID and values as list of
                relevant documents.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch service.

        Returns:
            X: List of feature vectors extracted for each pair of query and
                retrieved or relevant document.
            y: List of corresponding labels.
    '''
    X, Y = [], []
    for query_id in query_ids:

        query = all_queries[query_id]
        query_terms = analyze_query(es, query, 'body', index=index)

        valid_query = len(query_terms) > 0
        if not valid_query:
            continue

        top100baseline = es.search(
            q=query_terms,
            index=index,
            size=100,
            stored_fields=[]
        )['hits']['hits']
        top100ids = [doc['_id'] for doc in top100baseline]

        relevant_documents = all_qrels[query_id]
        doc_ids = unique_ids(relevant_documents, top100ids)
        relevant_documents_set = set(relevant_documents)

        for doc_id in doc_ids:
            label = 1 if doc_id in relevant_documents_set else 0
            feature_vector = extract_features(
                query_terms, doc_id, es, index
            )
            X.append(feature_vector)
            Y.append(label)

    return X, Y

class PointWiseLTRModel:
    def __init__(self) -> None:
        '''Instantiates LTR model with an instance of scikit-learn regressor.'''
        self.regressor = LinearRegression()

    def _train(self, X: List[List[float]], y: List[float]) -> None:
        '''Trains an LTR model.

        Args:
            X: Features of training instances.
            y: Relevance assessments of training instances.
        '''
        assert self.regressor is not None
        self.model = self.regressor.fit(X, y)

    def rank(
        self, ft: List[List[float]], doc_ids: List[str]
    ) -> List[Tuple[str, float]]:
        '''Predicts relevance labels and rank documents for a given query.

        Args:
            ft: A list of feature vectors for query-document pairs.
            doc_ids: A list of document ids.
        Returns:
            List of tuples, each consisting of document ID and predicted
                relevance label.
        '''
        assert self.model is not None
        rel_labels = self.model.predict(ft)
        sort_indices = np.argsort(rel_labels)[::-1]
        results = []
        for i in sort_indices:
            results.append((doc_ids[i], rel_labels[i]))
        return results


def get_rankings(
    ltr: PointWiseLTRModel,
    query_ids: List[str],
    all_queries: Dict[str, str],
    es: Elasticsearch,
    index: str,
    rerank: bool = False,
) -> Dict[str, List[str]]:
    '''Generate rankings for each of the test query IDs.

    Args:
        ltr: A trained PointWiseLTRModel instance.
        query_ids: List of query IDs.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.
        rerank: Boolean flag indicating whether the first-pass retrieval
            results should be reranked using the LTR model.

    Returns:
        A dictionary of rankings for each test query ID.
    '''

    test_rankings = {}
    for i, query_id in enumerate(query_ids):
        print(
            'Processing query {}/{} ID {}'.format(
                i + 1, len(query_ids), query_id
            )
        )
        # First-pass retrieval
        query_terms = analyze_query(
            es, all_queries[query_id], 'body', index=index
        )
        if len(query_terms) == 0:
            print(
                'WARNING: query {} is empty after analysis; ignoring'.format(
                    query_id
                )
            )
            continue
        hits = es.search(
            index=index, q=' '.join(query_terms), _source=True, size=100
        )['hits']['hits']
        test_rankings[query_id] = [hit['_id'] for hit in hits]

        # Rerank the first-pass result set using the LTR model.
        if rerank:
            feature_vectors = [
                extract_features(query_terms, doc_id, es, index=index) 
                for doc_id in test_rankings[query_id]
            ]
            updated_ranks = ltr.rank(feature_vectors, test_rankings[query_id])
            test_rankings[query_id] = [rank[0] for rank in updated_ranks]

    return test_rankings


def get_reciprocal_rank(
    system_ranking: List[str], ground_truth: List[str]
) -> float:
    '''Computes Reciprocal Rank (RR).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        RR (float).
    '''
    for i, doc_id in enumerate(system_ranking):
        if doc_id in ground_truth:
            return 1 / (i + 1)
    return 0


def get_mean_eval_measure(
    system_rankings: Dict[str, List[str]],
    ground_truths: Dict[str, Set[str]],
    eval_function: Callable,
) -> float:
    '''Computes a mean of any evaluation measure over a set of queries.

    Args:
        system_rankings: Dict with query ID as key and a ranked list of document
            IDs as value.
        ground_truths: Dict with query ID as key and a set of relevant document
            IDs as value.
        eval_function: Callback function for the evaluation measure that mean is
            computed over.

    Returns:
        Mean evaluation measure (float).
    '''
    sum_score = 0
    for query_id, system_ranking in system_rankings.items():
        sum_score += eval_function(system_ranking, ground_truths[query_id])
    return sum_score / len(system_rankings)

if __name__ == '__main__':
# Indexing documents to index=trec9_index
    # print('Resetting index ...')
    es = Elasticsearch(timeout=120)
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
    # print('Indexing documents ...')
    index_documents(DATA_PATH.format('documents.jsonl'), es, INDEX_NAME)


# # Testing
#     import random
#     def train_test_split(queries):
#         random.seed(a=1234567)
#         query_ids = sorted(list(queries.keys()))
#         random.shuffle(query_ids)
#         train_size = int(len(query_ids) * 0.8)
#         return query_ids[:train_size], query_ids[train_size:][-100:]

#     print('Loading queries and query-relevant document ids ...')
#     queries = load_queries(DATA_PATH.format('queries'))
#     qrels = load_qrels(DATA_PATH.format('qrels'))
    
#     train, test = train_test_split(queries)

#     print('Preparing training data ...  (might take a while)')
#     X_train, y_train = prepare_ltr_training_data(
#         train[:800], queries, qrels, es, index=INDEX_NAME
#     )
    
#     print('Training model ...')
#     # Instantiate PointWiseLTRModel.
#     ltr = PointWiseLTRModel()
#     ltr._train(X_train, y_train)

#     print('Ranking documents based on first-pass ...')
#     first_pass_ranks = get_rankings(
#         ltr, test, queries, es, index=INDEX_NAME, rerank=False
#     )

#     print('Ranking with re-ranking ...')
#     re_ranked_ranks = get_rankings(
#         ltr, test, queries, es, index=INDEX_NAME, rerank=True
#     )

#     print('-' * 30, ' Results ', '-' * 30)
#     print('First-pass (MRR):')
#     print(get_mean_eval_measure(first_pass_ranks, qrels, get_reciprocal_rank))
#     print('Re-ranked (MRR):')
#     print(get_mean_eval_measure(re_ranked_ranks, qrels, get_reciprocal_rank))
#     print('-' * 69)