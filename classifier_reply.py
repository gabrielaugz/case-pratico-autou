# classifier_responder.py
import openai
import os

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
                        "Você é um classificador de e-mails. "
                        "Responda apenas com 'Produtivo' ou 'Improdutivo'.\n\n"

                        "Definições:\n"
                        " - Produtivo: O email requer uma ação ou resposta específica (ex.: pedido de informação, solicitação de suporte).\n"
                        " - Improdutivo: O email não requer ação, como agradecimentos, felicitações, ou não há pedido.\n\n"

                        "Exemplos:\n"
                        "1) 'Olá, podem me enviar o relatório mensal?' => Produtivo\n"
                        "2) 'Obrigado pela ajuda ontem!' => Improdutivo\n"
                        "3) 'Olá, quero saber o status do meu pedido #12345' => Produtivo\n"
                        "4) 'Bom dia, só queria desejar boas festas!' => Improdutivo\n\n"
                        "Analise o email a seguir e retorne APENAS 'Produtivo' ou 'Improdutivo'."
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

        # padrão: ser produtivo
        if classification not in ["produtivo", "improdutivo"]:
            classification = "Produtivo"
        return classification.capitalize()

    except Exception as e:
        print("Erro:", e)
        return "Erro"

def suggest_reply(category: str, email_text: str) -> str:
    """
    Usa OpenAI GPT-3.5 para gerar uma resposta automática.
    """
    try:
        if category.lower() == "produtivo":
            prompt = f"Escreva uma breve resposta profissional para este email, indicando que estamos analisando a solicitação e retornaremos em breve:\n\n{email_text}"
        else:
            prompt = f"Escreva uma breve resposta amigável para este email, agradecendo a mensagem e informando que não há pendências:\n\n{email_text}"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente virtual que deve redigir respostas curtas em português."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print("Erro ao chamar OpenAI para geração de resposta:", e)
        return "Não foi possível gerar resposta."
