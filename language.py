import re
import string
import unidecode
from collections import Counter

import numpy as np
import pandas as pd

import nltk
from nltk import ngrams
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('wordnet')


#Function to remove punctuation
def remove_punct(text, punctuation=string.punctuation):
    text_nopunct = "".join([char for char in text if char not in punctuation])
    return text_nopunct

#Function to remove accents:
def strip_accents(text):
    text_noaccents = "".join([unidecode.unidecode(char) for char in text])
    return text_noaccents

# Function to tokenize words
def tokenize(text, to_lowercase=True):
    if to_lowercase:
        tokens = re.split(r'\W+', text.lower())
    else:
        tokens = re.split(r'\W+', text)
    #W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.
    return tokens

# Function to tokenize n_grams words
def tokenize_ngrams(text, ngram_range=(1, 1), to_lowercase=True):
    tokens = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        n_grams = ngrams(text.split(), n)
        for grams in n_grams:
            if to_lowercase:
                tokens.append(" ".join(grams).lower())
            else:
                tokens.append(" ".join(grams))
    return tokens

# Function to remove stopwords
def remove_stopwords(tokenized_text, stopwords=stopwords.words('portuguese')):
    text = [word for word in tokenized_text if word not in stopwords]
    return text

# Function to stemm tokens
def stemming(tokenized_text, stemmer=nltk.stem.RSLPStemmer()):
    text = [stemmer.stem(word) for word in tokenized_text if word not in ['']]
    return text

# Function to lemmatize tokens
def lemmatizing(tokenized_text, lemmatizer=nltk.WordNetLemmatizer()):
    text = [lemmatizer.lemmatize(word) for word in tokenized_text if word not in ['']]
    return text


# Preprocessor implementation
def preprocess_text(text, to_lowercase=True, punctuation=string.punctuation, strip_accents=True):
    # removing punctuation and converting to lower case:
    if to_lowercase:
        text = "".join([word.lower() for word in text if word not in punctuation])
    else:
        text = "".join([word for word in text if word not in punctuation])
    # removing accents:
    if strip_accents:
        text = "".join([unidecode.unidecode(word) for word in text])
    return text

# Analyzer implementation
def clean_text_stem(text, to_lowercase=True, punctuation=string.punctuation, strip_accents=True,
                    stemmer=nltk.stem.RSLPStemmer(), stopwords=stopwords.words('portuguese'), ngram_range = (1, 3)):
    # Text cleaning:
    clean_text = preprocess_text(text, to_lowercase, punctuation, strip_accents)
    if strip_accents:
        stopwords = [unidecode.unidecode(word) for word in stopwords]
    # Tokenization:
    tokenized_text = tokenize_ngrams(clean_text, ngram_range = ngram_range, to_lowercase=to_lowercase)
    # Pre-processing:
    corpus = [stemmer.stem(word) for word in tokenized_text if word not in stopwords+['']]
    return corpus

def clean_text_lemmatize(text, to_lowercase=True, punctuation=string.punctuation, strip_accents=True,
                         lemmatizer=nltk.WordNetLemmatizer(), stopwords=stopwords.words('portuguese'), ngram_range = (1, 3)):
    # Text cleaning:
    clean_text = preprocess_text(text, to_lowercase, punctuation, strip_accents)
    if strip_accents:
        stopwords = [unidecode.unidecode(word) for word in stopwords]
    # Tokenization:
    tokenized_text = tokenize_ngrams(clean_text, ngram_range = ngram_range, to_lowercase=to_lowercase)
    # Pre-processing:
    corpus = [lemmatizer.lemmatize(word) for word in tokenized_text if word not in stopwords+['']]
    return corpus


#### Feature engineering functions ###
def count_punctuation(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

def get_text_len(text_list):
    text_len = [len(x) for x in text_list]
    return text_len

# Text Vectorization:
def vectorize_column(df_train, df_test, col, vectorizer, vectorizer_name=''):
  corpus = df_train[col].values
  vectorizer.fit(corpus)

  df_train_vec = vectorizer.transform(df_train[col])
  vect_cols = [f'{vectorizer_name}_{col}_{i}' for i in range(df_train_vec.shape[1])]
  df_train_vec = pd.DataFrame(df_train_vec.toarray(), columns=vect_cols)
  df_train = pd.concat([df_train, df_train_vec], axis=1)
  df_test_vec = vectorizer.transform(df_test[col])
  df_test_vec = pd.DataFrame(df_test_vec.toarray(), columns=vect_cols)
  df_test = pd.concat([df_test, df_test_vec], axis=1)

  return df_train, df_test, vect_cols

# Words Essence:
class WordsEssence:
    def __init__(self, max_features=10, ngram_range=(1, 1), stopwords=stopwords.words('portuguese')):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stopwords = stopwords
        self.essence_tokens = []

    def fit(self, x_array):
        all_tokens = []
        for sentence in x_array:
            all_tokens += [word for word in tokenize_ngrams(strip_accents(remove_punct(sentence)), self.ngram_range) if word not in self.stopwords]
        counts = Counter(all_tokens)
        self.essence_tokens = [word_count[0] for word_count in counts.most_common(self.max_features)]
        return self

    def transform(self, x_array):
        essence_index = []
        for sentence in x_array:
            tokenized_text = [word for word in tokenize_ngrams(strip_accents(remove_punct(sentence)), self.ngram_range) if word not in self.stopwords]
            idx_val = (np.isin(tokenized_text, self.essence_tokens).sum() / len(self.essence_tokens)) ** (1/2)
            essence_index.append(idx_val)
        return essence_index
