import json
import math
from operator import itemgetter
from typing import Any, Dict, List, Union

from elasticsearch import Elasticsearch
from numpy import indices

TYPE_PREDICATE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
NAME_PREDICATES = set(
    [
        'http://www.w3.org/2000/01/rdf-schema#label',
        'http://xmlns.com/foaf/0.1/name',
        'http://xmlns.com/foaf/0.1/givenName',
        'http://xmlns.com/foaf/0.1/surname',
    ]
)
TYPE_PREDICATES = set([TYPE_PREDICATE, 'http://purl.org/dc/terms/subject'])
COMMENT_PREDICATE = 'http://www.w3.org/2000/01/rdf-schema#comment'


INDEX_NAME = 'musicians'
INDEX_SETTINGS = {
    'mappings': {
        'properties': {
            'names': {
                'type': 'text',
                'term_vector': 'yes',
                'analyzer': 'english',
            },
            'description': {
                'type': 'text',
                'term_vector': 'yes',
                'analyzer': 'english',
            },
            'attributes': {
                'type': 'text',
                'term_vector': 'yes',
                'analyzer': 'english',
            },
            'related_entities': {
                'type': 'text',
                'term_vector': 'yes',
                'analyzer': 'english',
            },
            'types': {
                'type': 'text',
                'term_vector': 'yes',
                'analyzer': 'english',
            },
            'catch_all': {
                'type': 'text',
                'term_vector': 'yes',
                'analyzer': 'english',
            },
        }
    }
}

def has_type(properties: Dict[str, Any], target_type: str) -> bool:
    '''Check whether properties contain specific type

    Args:
        properties: Dictionary with properties
        target_type: Type to check for.

    Returns:
        True if target type in properties.
    '''
    if TYPE_PREDICATE not in properties:
        return False
    for p in properties[TYPE_PREDICATE]:
        if p['value'] == target_type:
            return True
    return False


def resolve_uri(uri: str) -> str:
    '''Resolves uri.'''
    uri = uri.split('/')[-1].replace('_', ' ')
    if uri.startswith('Category:'):
        uri = uri[len('Category:') :]
    return uri

def dict_based_entity(entity_properties: Dict[str, Any]) -> Dict[str, Union[str, List]]:
    '''Create a simpler dictionary-based entity representation.

    Args:
        entity_id: The ID of the entity.

    Returns:
        A dictionary-based entity representation.
    '''
    dber = {}
    for uri, info in entity_properties.items():
        values = []
        for info_dict in info:
            val = info_dict.get('value', None)
            if val is not None:
                values.append(val)
        dber[uri] = values

    # Make 1-length-lists into single strings instead
    dber = {k:v[0] if (len(v) == 1) else v for (k,v) in dber.items()}
    return dber

def fielded_doc_entity(entity_properties: Dict[str, Any]) -> Dict[str, str]:
    '''fielded document representation should include the fields `names`,
    `description`, `attributes`, `related_entities`, `types`, and `catch_all`.


    They should contain the following:
        * `names`: the objects of `NAME_PREDICATES`,
        * `description`: the object(s) of `COMMENT_PREDICATE`, 
        * `attributes`: objects that are literal values, 
        * `related_entities`: objects that are entities, 
        * `types`: the objects of `TYPE_PREDICATES`, and
        * `catch_all`: all of the above.

    NB! All fields except `catch_all` are mutually exclusive.

    Args:
        entity_id: The ID of the entity.
        **kwargs: Additional keyword arguments. Notably, session to provide to
            get_dbpedia_entity() function

    Returns:
        Dictionary with the above stated keys.
    '''
    field_doc = {
        'names': '',               # the objects of `NAME_PREDICATES`,
        'description': '',         # the object(s) of `COMMENT_PREDICATE`, 
        'attributes': '',          # objects that are literal values, 
        'related_entities': '',    # objects that are entities, 
        'types': '',               # the objects of `TYPE_PREDICATES`, and
        'catch_all': '',           # all of the above.
    }

    for uri, info in entity_properties.items():

        vals = [resolve_uri(str(v['value'])) if v['type'] == 'uri'
                else str(v['value']) for v in info]
        # Adding a space at the end in order to seperate ending this 
        # iteration's last word from the first word in next iteration which 
        # adds to the same field
        string_vals = ' '.join(vals) + ' '
        field_doc['catch_all'] += string_vals

        if uri in NAME_PREDICATES:
            field_doc['names'] += string_vals

        elif uri in COMMENT_PREDICATE:
            field_doc['description'] += string_vals

        elif uri in TYPE_PREDICATES:
            field_doc['types'] += string_vals

        else:
            types = [v['type'] for v in info]

            literal_string = (
                ' '.join(v for (v, t) in zip(vals, types) if (t == 'literal'))
            ) + ' '

            related_entities_string = (
                ' '.join(v for (v, t) in zip(vals, types) if (t == 'uri'))
            ) + ' '
 
            field_doc['attributes'] += literal_string
            field_doc['related_entities'] += related_entities_string

    return field_doc


def bulk_index(es: Elasticsearch, artists: Dict[str, Any]) -> None:
    '''Iterate over artists, and index those that are of the
    right type.

    Args:
        es: Elasticsearch instance.
        artists: Dictionary with artist names and their properties.
    '''
    artist_type = 'http://dbpedia.org/ontology/MusicalArtist'

    for art, art_prop in artists.items():
        # Check if artist have type: 'MusiaclArtist'
        if not has_type(art_prop, artist_type):
            continue
        art_fields = fielded_doc_entity(art_prop)
        es.index(index=INDEX_NAME, id=art, document=art_fields)

       

def baseline_retrieval(
    es: Elasticsearch, index_name: str, query: str, k: int = 100
) -> List[str]:
    '''Performs baseline retrival on index.

    Args:
        es: Elasticsearch instance.
        index_name: A string of text.
        query: A string of text, space separated terms.
        k: An integer.

    Returns:
        A list of entity IDs as strings, up to k of them, in descending order of
            scores.
    '''
    search_results = es.search(
        index=index_name, body={'query': {'match': {'catch_all': query}}}, 
        _source=False, size=k
    )
    
    scores = [(r['_id'], r['_score']) for r in search_results['hits']['hits']]
    # Pythons Timsort algorithm is stable, meaning items will preserve the same
    # order unless the key to be sorted on is the same. This lets us sort twice
    # First on the non-primary key (id), then on the primary key (score):
    scores = sorted(scores, key=lambda id_score: id_score[0], reverse=True)
    scores = sorted(scores, key=lambda id_score: id_score[1], reverse=True)
    
    return [id_scores[0] for id_scores in scores]


def analyze_query(es: Elasticsearch, query: str) -> List[str]:
    '''Analyzes query and returns a list of query terms that can be found in
    the collection.

    Args:
        es: Elasticsearch instance
        query: Query to analyze

    Returns:
        List of query terms.
    '''
    tokens = es.indices.analyze(
        index=INDEX_NAME, body={'text': query, 'analyzer': 'english'}
    )['tokens']
    query_terms = []
    for t in sorted(tokens, key=lambda x: x['position']):
        query_terms.append(t['token'])
    return query_terms


class CollectionLM:
    def __init__(
        self, es: Elasticsearch, qterms: List[str], fields: List[str] = None,  # type: ignore
    ) -> None:
        '''This class is used for obtaining collection language modeling
        probabilities $P(t|C_i)$.

        Args:
            es: Elasticsearch instance
            qterms: List of query terms
            fields: List of entity fields
        '''
        self._es = es
        self._probs = {}
        self._fields = fields or [
            'names',
            'description',
            'attributes',
            'related_entities',
            'types',
            'catch_all',
        ]
        # computing P(t|C_i) for each field and for each query term
        for field in self._fields:
            self._probs[field] = {}
            for t in qterms:
                self._probs[field][t] = self._get_prob(field, t)

    @property
    def fields(self):
        return self._fields

    def _get_prob(self, field: str, term: str) -> float:
        '''computes the collection Language Model probability of a term for a
        given field.

        Args:
            field: Fields for which to get the probability
            term: Term for which to get the probability

        Returns:
            Collection LM probability.
        '''
        # use a boolean query to find a document that contains the term
        hits = (
            self._es.search(
                index=INDEX_NAME,
                query={'match': {field: term}},
                _source=False,
                size=1,
            )
            .get('hits', {})
            .get('hits', {})
        )
        doc_id = hits[0]['_id'] if len(hits) > 0 else None
        
        # ask for global term statistics when requesting the term vector of
        # that doc (`term_statistics=True`)
        
        prob = 0.0

        if doc_id:  # there exists a document having this ttf
            doc = self._es.termvectors(index=INDEX_NAME, 
                                id=doc_id, 
                                fields=field,
                                term_statistics=True
            )
            ttf = (
                doc['term_vectors'][field]['terms']
                    .get(term, {})
                    .get('ttf', 0)
            )
            sum_ttf = (
                doc['term_vectors'][field]['field_statistics']['sum_ttf']
            )
        
            if not sum_ttf == 0:
                prob = ttf / sum_ttf
            
        return prob

    def prob(self, field: str, term: str) -> float:
        '''Return probability for a given field and term.
        
        Args:
            field: Fields for which to get the probability
            term: Term for which to get the probability

        Returns:
            Collection LM probability.
        '''
        return self._probs.get(field, {}).get(term, 0)


def get_term_mapping_probs(clm: CollectionLM, term: str) -> Dict[str, float]:
    '''PRMS: For a single term, find their mapping probabilities for all fields.

    Args:
        clm: Collection language model instance.
        term: A single-term string.

    Returns:
        Dictionary of mapping probabilities for the fields.
    '''
    mapping_probs = {}
    for field in clm.fields:
        p_tf = clm._probs[field].get(term, 0) # prob of term in field
        P_f = 1.0 / (len(clm.fields)) # priori of field
        P_t = sum(clm._probs[f].get(term, 0) for f in clm.fields) * P_f
        
        if P_t == 0:  # p_tf is also 0 and probability is 0
            mapping_probs[field] = 0
        else:
            mapping_probs[field] = p_tf * P_f / P_t

    return mapping_probs

def score_prms(
    es, clm: CollectionLM, qterms: List[str], doc_id: str, mu: int = 100
):
    '''Score PRMS.'''
    # Getting term frequency statistics for the given document field from
    # Elasticsearch
    # Note that global term statistics are not needed (`term_statistics=False`)
    tv = es.termvectors(
        index=INDEX_NAME, id=doc_id, fields=clm.fields, term_statistics=False
    ).get('term_vectors', {})

    # compute field lengths $|d_i|$
    len_d_i = []  # document field length
    for i, field in enumerate(clm.fields):
        if field in tv:
            len_d_i.append(
                sum([s['term_freq'] for _, s in tv[field]['terms'].items()])
            )
        else:  # that document field may be empty
            len_d_i.append(0)

    # scoring the query
    score = 0  # log P(q|d)
    for t in qterms:
        Pt_theta_d = 0  # P(t|\theta_d)
        # Get field mapping probs.
        Pf_t = get_term_mapping_probs(clm, t)
        for i, field in enumerate(clm.fields):
            if field in tv:
                ft_di = (
                    tv[field]['terms'].get(t, {}).get('term_freq', 0)
                )  # $f_{t,d_i}$
            else:  # that document field is empty
                ft_di = 0
            Pt_Ci = clm.prob(field, t)  # $P(t|C_i)$
            Pt_theta_di = (ft_di + mu * Pt_Ci) / (
                mu + len_d_i[i]
            )  # $P(t|\theta_{d_i})$ with Dirichlet smoothing
            Pt_theta_d += Pf_t[field] * Pt_theta_di
        score += math.log(Pt_theta_d)

    return score


def prms_retrieval(es: Elasticsearch, query: str):
    '''PRMS retrieval.'''
    # Analyze query
    query_terms = analyze_query(es, query)

    # Perform initial retrieval using ES
    res = es.search(
        index=INDEX_NAME, q=query, df='catch_all', _source=False, size=200
    ).get('hits', {})

    # Instantiate collectionLM class
    clm = CollectionLM(es, query_terms)

    # Rerank results using PRMS
    scores = {}
    for doc in res.get('hits', {}):
        doc_id = doc.get('_id')
        scores[doc_id] = score_prms(es, clm, query_terms, doc_id)

    return [
        x[0] for x in sorted(scores.items(), key=itemgetter(1, 0), reverse=True)
    ]


def reset_index(es: Elasticsearch) -> None:
    '''Reset Index'''
    if es.indices.exists(INDEX_NAME):  # type: ignore
        es.indices.delete(index=INDEX_NAME)

    es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)

def get_artists() -> Dict[str, Any]:
    '''Loads artists from file.'''
    with open('data/artists.json', 'r') as f:
        return json.load(f)

def main():
    '''Index artists'''
    es = Elasticsearch()
    es.info()

    artists = get_artists()

    reset_index(es)
    bulk_index(es, artists)


if __name__ == '__main__':
    main()

