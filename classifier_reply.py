# classifier_responder.py
import openai
import os
from models.ml_classifier import classify_email_ml
from models.ml_reply_generator import generate_reply_ml

# chave da api via var de ambiente
openai.api_key = os.getenv("OPENAI_API_KEY", "")

def classify_email(email_text: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente que classifica e-mails recebidos. "
                        "Analise o e-mail a seguir e classifique **somente como 'Produtivo' ou 'Improdutivo'**.\n\n"

                        "**Definições:**\n"
                        "- **Produtivo**: O email contém uma solicitação direta, pergunta ou tarefa que exige uma resposta ou ação específica. "
                        "Exemplos incluem: pedidos de status, requisição de informações, suporte técnico ou solicitações comerciais.\n"
                        "- **Improdutivo**: O email não requer uma ação específica ou não contribui para o fluxo de trabalho. "
                        "Exemplos incluem: agradecimentos, mensagens de boas festas, e-mails automáticos ou informações irrelevantes.\n\n"

                        "**Exemplos:**\n"
                        "1️. 'Olá, podem me enviar o relatório mensal?' = Produtivo**\n"
                        "2️. 'Obrigado pela ajuda ontem!' = Improdutivo**\n"
                        "3️. 'Olá, quero saber o status do meu pedido #12345' = Produtivo**\n"
                        "4️. 'Bom dia, só queria desejar boas festas!' = Improdutivo**\n\n"

                        "**IMPORTANTE:** Retorne **apenas** a palavra 'Produtivo' ou 'Improdutivo'. Nenhuma outra palavra!"
                    )
                },
                {
                    "role": "user",
                    "content": email_text
                }
            ],
            max_tokens=5,
            temperature=0
        )

        classification = response.choices[0].message.content.strip().lower()

        # caso OpenAI erre a classificaçao, utiliza-se o modelo de ML
        if classification not in ["produtivo", "improdutivo"]:
            print("[INFO] OpenAI classificou de forma incerta. Utilizando ML...")
            classification = classify_email_ml(email_text)
        
        return classification.capitalize()

    except openai.error.OpenAIError as e:
        print(f"[ERROR] OpenAI falhou ({e}). Usando Machine Learning tradicional.")
        return classify_email_ml(email_text)

    except Exception as e:
        print(f"[ERROR] Erro inesperado ({e}). Usando Machine Learning tradicional.")
        return classify_email_ml(email_text)

def suggest_reply(category: str, email_text: str) -> str:
    """
    Usa OpenAI GPT-3.5 para gerar uma resposta automática.
    """
    try:
        if category.lower() == "produtivo":
            prompt = (
                "Escreva uma resposta profissional e curta para este email. "
                "Confirme o recebimento e informe que analisaremos e retornaremos em breve. "
                "Se relevante, peça mais informações para agilizar a resposta.\n\n"
                f"Email: {email_text}\n\n"
                "Resposta:"
            )
        else:
            prompt = (
                "Escreva uma resposta amigável e breve para este email. "
                "Agradeça a mensagem e informe que não há pendências para tratar.\n\n"
                f"Email: {email_text}\n\n"
                "Resposta:" 
            )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente virtual que deve redigir respostas curtas em português."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return answer
    
    except openai.error.OpenAIError as e:
        print(f"[ERROR] OpenAI falhou ({e}). Usando Machine Learning tradicional para resposta.")
        return generate_reply_ml(email_text)

    except Exception as e:
        print(f"[ERROR] Erro inesperado ({e}). Usando Machine Learning tradicional para resposta.")
        return generate_reply_ml(email_text)
