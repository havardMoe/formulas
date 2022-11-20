from itertools import count
from typing import Dict, List
from collections import Counter
import string

def get_word_frequencies(doc: str) -> Dict[str, int]:
    """Extracts word frequencies from a document.

    Args:
        doc: Document content given as a string.

    Returns:
        Dictionary with words as keys and their frequencies as values.
    """

    # Decided not to use string.punctuation because assignment text said
    # specifically to include: [",", ".", ":", ";", "?", "!"].
    for c in  [",", ".", ":", ";", "?", "!"]:
        doc = doc.replace(c, " ")

    # Make all words lowercase
    # Remove whitespace and split text into words
    # Count word frequencies 
    counts = Counter(doc.lower().split)
    return counts


def get_word_feature_vector(
    word_frequencies: Dict[str, int], vocabulary: List[str]
) -> List[int]:
    """Creates a feature vector for a document, comprising word frequencies
        over a vocabulary.

    Args:
        word_frequencies: Dictionary with words as keys and frequencies as
            values.
        vocabulary: List of words.

    Returns:
        List of length `len(vocabulary)` with respective frequencies as values.
    """
    return [word_frequencies.get(word, 0) for word in vocabulary]