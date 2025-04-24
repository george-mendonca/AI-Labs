# app.py

import os
from flask import Flask, request, render_template, redirect, url_for, session
from dotenv import load_dotenv
import google.generativeai as genai

# Carrega a chave da API do arquivo .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Cria a aplicação Flask
app = Flask(__name__)
app.secret_key = "segredo"  # Necessário para usar sessões de usuário

# Modelos disponíveis (pode expandir)
MODELOS = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash"
]

@app.route("/", methods=["GET", "POST"])
def chat():
    resposta = None
    prompt = ""

    if request.method == "POST":
        # Botão "Limpar" apaga dados da sessão
        if "limpar" in request.form:
            session.pop("last_prompt", None)
            session.pop("last_response", None)
            session.pop("modelo", None)
            return redirect(url_for("chat"))

        # Lê entrada do usuário e o modelo selecionado
        prompt = request.form.get("prompt", "").strip()
        modelo = request.form.get("modelo", MODELOS[0])
        session["last_prompt"] = prompt
        session["modelo"] = modelo

        if prompt:
            try:
                # Cria o modelo selecionado
                model = genai.GenerativeModel(modelo)
                # Envia o prompt para a API
                response = model.generate_content(prompt)
                # Salva a resposta e exibe
                resposta = response.text.strip()
                session["last_response"] = resposta
            except Exception as e:
                resposta = f"Erro: {str(e)}"
                session["last_response"] = resposta
        else:
            resposta = None

    else:
        # Em caso de GET/reload, limpa tudo
        session.pop("last_prompt", None)
        session.pop("last_response", None)
        prompt = ""
        resposta = None

    modelo = session.get("modelo", MODELOS[0])

    # Renderiza o HTML com os dados
    return render_template("index.html", resposta=resposta, modelos=MODELOS, prompt=prompt, modelo_atual=modelo)

# Inicia o servidor Flask local
if __name__ == "__main__":
    app.run(debug=True)