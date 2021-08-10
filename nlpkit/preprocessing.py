# preprocessing.py

import re
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict

EMAIL_PATTERN = r'[A-Za-z0-9_.-]+@[A-Za-z]+.[A-Za-z]{,3}'
YEARS_OF_EXP_PATTERN = r'\d{1,}\+?\s?Years'

def preprocess_ents(label, text):
    """
    Preprocess an entity

    Parameters
    ----------
    label: str:
        The named entity label
    text: str:
        The name entity text

    Returns
    -------
    tuple: returns a tuple of strings containing the label and text 
    """

    pattern = ':'
    match = re.search(pattern, text)
    if match:
        text = text[match.end():]
    text = re.sub('^\s+', '', text)
    text = re.sub('[^A-Za-z0-9._,@]+', ' ', text) # remove non-alphanumeric characters
    
    return label, text

def extract_ents(doc):
    """
    Extracts named entity from a document

    Parameters
    ----------
    doc: object
        spaCy Doc object

    returns
    ------
    dict: returns a dictionary of named entities
    """

    ents = defaultdict(list)
    for ent in doc.ents:
        label, text = preprocess_ents(ent.label_.lower(), ent.text)
        if not text in ents[label]: # store only unique entities
            ents[label].append(text)
            
    # extract email if availaible
    emails = re.findall(EMAIL_PATTERN, doc.text)
    if len(emails) > 0:
        for email in emails:
            ents['email address'].append(email)
    
    # extract years of experience if available
    years_of_experience = re.search(YEARS_OF_EXP_PATTERN, doc.text, re.IGNORECASE)
    
    if years_of_experience:
        ents['years of experience'].append(years_of_experience.group())
        
    return dict(ents)

def remove_accented_chars(text):
    """
    Removes accented characters from a string object

    Parameters
    ----------
    text: str
        String object

    Returns
    -------
        str: returns a string object

    """

    text = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8')
    return text

def remove_stopwords(text):
    """
    Removes stop words from a string object

    Parameters
    ----------
    text: str
        String object

    Returns
    -------
        str: returns a string object

    """

    return ' '.join([t for t in text.split() if t.lower() not in STOP_WORDS])


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    DataPreprocessor()

    Preprocess a document

    DataPreprocessor applies various preprocessing technique to a text document

    Examples
    ----------
    >>> import numpy as np
    >>> from nlpkit.preprocessing import DataPreprocessor
    >>> dp = DataPreprocessor()
    >>> X = np.array([['I am a boy'], ['I am a girl']])
    >>> dp.transform(X)

    """

    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # remove puntuations 
        pattern = r'[^a-z A-Z 0-9 -]+'
        X.iloc[:, 0] = X.iloc[:, 0].apply(lambda x: re.sub(pattern, '', x))
        
        # remove multiple spaces
        X.iloc[:, 0] = X.iloc[:, 0].apply(lambda x: " ".join(x.split()))
        
        # convert to lower case
        X.iloc[:, 0] = X.iloc[:, 0].apply(lambda x: x.lower())
        
        # remove accented characters
        X.iloc[:, 0] = X.iloc[:, 0].apply(lambda x: remove_accented_chars(x))

        # remove stop words
        X.iloc[:, 0] = X.iloc[:, 0].apply(lambda x: remove_stopwords(x))
        
        return X.values[:, 0]