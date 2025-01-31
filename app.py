# app.py

from flask import Flask, render_template, request, redirect, url_for
import os
import PyPDF2 
from classifier_reply import classify_email, suggest_reply

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' 

# rota principal (get) 
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# rota para processar o texto digitado no <textarea>
@app.route('/process_text', methods=['POST'])
def process_text():
    email_text = request.form.get('email_text', '')

    if not email_text.strip():
        return redirect(url_for('index'))

    category = classify_email(email_text)

    suggested = suggest_reply(category, email_text)

    return render_template('index.html', category=category, suggested_reply=suggested)

# rota para processar o upload de arquivo
@app.route('/process_file', methods=['POST'])
def process_file():
    file = request.files.get('email_file', None)

    if not file or file.filename == '':
        return redirect(url_for('index'))

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    email_text = ""

    if ext == '.txt':
        email_text = file.read().decode('utf-8', errors='ignore')
    elif ext == '.pdf':
        try:
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)

            if not pdf_reader.pages:
                return render_template('index.html', error="O PDF não contém páginas legíveis.")

            for page in pdf_reader.pages:
                email_text += page.extract_text() + "\n"

        except PyPDF2.errors.PdfReadError:
            return render_template('index.html', error="Erro: Arquivo PDF inválido ou corrompido.")

        except Exception as e:
            return render_template('index.html', error=f"Erro ao processar o PDF: {e}")

    else:
        return render_template('index.html', error="Formato de arquivo não suportado. Envie um arquivo .txt ou .pdf.")

    # classificação e geração de resposta
    category = classify_email(email_text)
    suggested = suggest_reply(category, email_text)

    return render_template('index.html', category=category, suggested_reply=suggested)

if __name__ == '__main__':
    app.run(debug=True)
