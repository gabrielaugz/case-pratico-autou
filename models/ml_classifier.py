# ml_classifier.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

nltk.download('stopwords')
portuguese_stopwords = stopwords.words('portuguese')

emails = [
    "Olá, poderiam me enviar o relatório?", "Preciso de um status do meu pedido #12345",
    "Obrigado pelo atendimento!", "Feliz Natal para toda a equipe!",
    "Segue em anexo o contrato assinado", "Enviei um pagamento, por favor confirmem o recebimento",
    "Só queria desejar um ótimo dia!", "Aguardo retorno sobre minha solicitação de reembolso",
]

labels = ["Produtivo", "Produtivo", "Improdutivo", "Improdutivo", 
          "Produtivo", "Produtivo", "Improdutivo", "Produtivo"]

# modelo para classificar com base em naive bayes
vectorizer = TfidfVectorizer(stop_words=portuguese_stopwords)
model = MultinomialNB()

pipeline = make_pipeline(vectorizer, model)
pipeline.fit(emails, labels)

def classify_email_ml(email_text: str) -> str:
    """
    Classifica um email usando o modelo de Machine Learning tradicional.
    """
    return pipeline.predict([email_text])[0]