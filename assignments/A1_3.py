# Types
from typing import List, Tuple, Union
from numpy import ndarray

# Data cleaning
import contractions
import string
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Feature extraction
from sklearn.feature_extraction.text import CountVectorizer

# Classification and evaluation
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# Prevent init of stemmer object each time the preprocessing function is called
LEMMATIZER = WordNetLemmatizer()


def load_data(path: str) -> Tuple[List[str], List[int]]:
    """Loads data from file. Each except first (header) is a datapoint
    containing ID, Label, Email (content) separated by "\t". Lables should be
    changed into integers with 1 for "spam" and 0 for "ham".

    Args:
        path: Path to file from which to load data

    Returns:
        List of email contents and a list of lobels coresponding to each email.
    """
    df = pd.read_csv(path, sep="\t", header=0)  
    # Represent labels as integers
    df["Label"] = df["Label"].apply(lambda lab: int(lab=="spam"))
    return list(df["Email"]), list(df["Label"]) # np.array didnt work (truth value ambigious test error)


def preprocess(doc: str) -> str:
    """Preprocesses text to prepare it for feature extraction.
    
    Args:
        doc: String comprising the unprocessed contents of some email file.

    Returns:
        String comprising the corresponding preprocessed text.
    """
    # Remove HTML
    html_pattern = re.compile("<[^>]+>")
    doc = html_pattern.sub(" ", doc)

    # Remove URL
    url_pattern = re.compile("http\S+")
    doc = url_pattern.sub(" ", doc)

    # Lowercase
    doc = doc.lower()

    # Expand contractions
    doc = contractions.fix(doc)

    # Remove punctuation for spaces
    punct_dict = {p: " " for p in string.punctuation}
    doc = doc.translate(str.maketrans(punct_dict))

    # Tokenize words and remove continuous words with a length above 25 chars
    tokens = [word for word in doc.split() if len(word) < 25]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]

    return " ".join(word for word in tokens)  # combine tokens to a doc


def preprocess_multiple(docs: List[str]) -> List[str]:
    """Preprocesses multiple texts to prepare them for feature extraction.

    Args:
        docs: List of strings, each consisting of the unprocessed contents
            of some email file.

    Returns:
        List of strings, each comprising the corresponding preprocessed
            text.
    """
    return [preprocess(doc) for doc in docs]


def extract_features(
    train_dataset: List[str], test_dataset: List[str]
) -> Union[Tuple[ndarray, ndarray], Tuple[List[float], List[float]]]:
    """Extracts feature vectors from a preprocessed train and test datasets.

    Args:
        train_dataset: List of strings, each consisting of the preprocessed
            email content.
        test_dataset: List of strings, each consisting of the preprocessed
            email content.

    Returns:
        A tuple of of two lists. The lists contain extracted features for 
          training and testing dataset respectively.
    """
    cv = CountVectorizer(max_features=40_000)
    # learn vocab from testset (no need to learn it from train set)
    cv.fit(test_dataset)
    test_vectors = cv.transform(test_dataset)
    train_vectors = cv.transform(train_dataset)
    return train_vectors, test_vectors 
    


def train(X: ndarray, y: List[int]) -> object:
    """Trains a classifier on extracted feature vectors.

    Args:
        X: Numerical array-like object (2D) representing the instances.
        y: Numerical array-like object (1D) representing the labels.

    Returns:
        A trained model object capable of predicting over unseen sets of
            instances.
    """
    # No need to fit prior for this project, as we are not going to predict
    # on live data
    mnb = MultinomialNB(fit_prior=False)
    mnb.fit(X, y)
    return mnb


def evaluate(y: List[int], y_pred: List[int]) -> Tuple[float, float, float, float]:
    """Evaluates a model's predictive performance with respect to a labeled
    dataset.

    Args:
        y: Numerical array-like object (1D) representing the true labels.
        y_pred: Numerical array-like object (1D) representing the predicted
            labels.

    Returns:
        A tuple of four values: recall, precision, F_1, and accuracy.
    """
    score_funcs = [
        metrics.recall_score,
        metrics.precision_score,
        metrics.f1_score,
        metrics.accuracy_score
    ]

    return [metric(y, y_pred) for metric in score_funcs]



if __name__ == "__main__":
    print("Loading data...")
    train_data_raw, train_labels = load_data("data/train.tsv")
    test_data_raw, test_labels = load_data("data/test.tsv")

    print("Processing data...")
    train_data = preprocess_multiple(train_data_raw)
    test_data = preprocess_multiple(test_data_raw)

    print("Extracting features...")
    train_feature_vectors, test_feature_vectors = extract_features(
        train_data, test_data
    )

    print("Training...")
    classifier = train(train_feature_vectors, train_labels)

    print("Applying model on test data...")
    predicted_labels = classifier.predict(test_feature_vectors)

    print("Evaluating")
    recall, precision, f1, accuracy = evaluate(test_labels, predicted_labels)

    print(f"Recall:\t{recall}")
    print(f"Precision:\t{precision}")
    print(f"F1:\t{f1}")
    print(f"Accuracy:\t{accuracy}")
