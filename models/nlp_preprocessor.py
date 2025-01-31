# nlp_preprocessor.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text(text: str) -> list:
    text = re.sub(r'[^a-zA-Z0-9áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ\s]', '', text)
    text = text.lower()

    tokens = nltk.word_tokenize(text, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))

    stemmer = PorterStemmer()
    processed_tokens = []
    for token in tokens:
        if token not in stop_words:
            stemmed_token = stemmer.stem(token)
            processed_tokens.append(stemmed_token)

    return processed_tokens