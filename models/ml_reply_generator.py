import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB

portuguese_stopwords = stopwords.words('portuguese')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "emails_replies.csv")

data = pd.read_csv(DATASET_PATH)
data.columns = data.columns.str.strip()

# modelo simples com naive bayes + tf-idf
vectorizer = TfidfVectorizer(stop_words=portuguese_stopwords)
classifier = MultinomialNB()

pipeline = make_pipeline(vectorizer, classifier)
pipeline.fit(data["Email"], data["Resposta"])

def generate_reply_ml(email_text: str) -> str:
    """
    Gera uma resposta automática utilizando Machine Learning.
    """
    return pipeline.predict([email_text])[0]
