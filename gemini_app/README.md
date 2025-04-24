
# 🤖 Tutorial Completo – Chat com IA (Google Gemini API + Flask)

Este projeto demonstra como criar uma aplicação web simples com **Python + Flask**, integrando com a **API do Google Gemini**, incluindo:

✅ Interface bonita com HTML/CSS  
✅ Suporte a múltiplos modelos (via combo)  
✅ Modo claro/escuro (toggle)  
✅ Sessão persistente com Flask  
✅ Comentários didáticos em todo o código  

---

## 📁 Estrutura do Projeto

```
chatgpt_gemini_app/
├── app.py
├── .env
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── style.css
```

---

## ✅ `app.py` – Aplicação principal

```python
# app.py

import os
from flask import Flask, request, render_template, redirect, url_for, session
from dotenv import load_dotenv
import google.generativeai as genai

# Carrega a chave da API salva no .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Inicializa o Flask
app = Flask(__name__)
app.secret_key = "segredo"  # Necessário para sessões

# Lista de modelos disponíveis
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
        # Se o usuário clicar em "Limpar", apagamos os dados salvos
        if "limpar" in request.form:
            session.pop("last_prompt", None)
            session.pop("last_response", None)
            session.pop("modelo", None)
            return redirect(url_for("chat"))

        # Lê o prompt e modelo do formulário
        prompt = request.form.get("prompt", "").strip()
        modelo = request.form.get("modelo", MODELOS[0])
        session["last_prompt"] = prompt
        session["modelo"] = modelo

        # Se houver prompt, envia à API
        if prompt:
            try:
                model = genai.GenerativeModel(modelo)
                response = model.generate_content(prompt)
                resposta = response.text.strip()
                session["last_response"] = resposta
            except Exception as e:
                resposta = f"Erro: {str(e)}"
                session["last_response"] = resposta
        else:
            resposta = None

    else:
        # Limpa ao dar reload (GET)
        session.pop("last_prompt", None)
        session.pop("last_response", None)
        prompt = ""
        resposta = None

    modelo = session.get("modelo", MODELOS[0])

    # Retorna o template com os dados
    return render_template("index.html", resposta=resposta, modelos=MODELOS, prompt=prompt, modelo_atual=modelo)

# Roda o servidor local
if __name__ == "__main__":
    app.run(debug=True)
```

---

## ✅ `templates/index.html` – HTML comentado

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="UTF-8">
  <title>Chat com Gemini</title>

  <!-- CSS -->
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">

  <!-- JS: loading e modo escuro -->
  <script>
    function mostrarLoading() {
      document.getElementById("loading").style.display = "block";
    }

    function toggleModo() {
      document.body.classList.toggle("dark");
    }
  </script>
</head>
<body>
  <div class="container">
    <h1>🤖 Chat com Gemini</h1>

    <!-- Form principal -->
    <form method="post" onsubmit="mostrarLoading()">
      <textarea name="prompt" rows="4" placeholder="Digite sua pergunta...">{{ prompt }}</textarea>

      <!-- Dropdown de modelo -->
      <select name="modelo">
        {% for m in modelos %}
          <option value="{{ m }}" {% if m == modelo_atual %}selected{% endif %}>{{ m }}</option>
        {% endfor %}
      </select>

      <!-- Botões -->
      <div class="botoes">
        <input type="submit" value="Enviar">
        <button name="limpar">Limpar Chat</button>
        <button type="button" onclick="toggleModo()">🌙/☀️</button>
      </div>
    </form>

    <!-- Indicador de carregamento -->
    <div id="loading">⏳ Gerando resposta...</div>

    <!-- Resposta da IA -->
    {% if resposta %}
      <div class="resposta">
        <h3>Resposta:</h3>
        <p>{{ resposta }}</p>
      </div>
    {% endif %}
  </div>
</body>
</html>
```

---

## ✅ `static/style.css` – Estilos com modo dark

```css
/* static/style.css */

body {
  font-family: 'Segoe UI', sans-serif;
  background: #f4f4f4;
  color: #222;
  margin: 0;
  padding: 2em;
  transition: all 0.4s;
}

body.dark {
  background: #121212;
  color: #eee;
}

.container {
  max-width: 700px;
  margin: auto;
  padding: 2em;
  background: white;
  border-radius: 12px;
  box-shadow: 0 0 10px #ccc;
}

body.dark .container {
  background: #1e1e1e;
  box-shadow: 0 0 10px #444;
}

textarea, select {
  width: 100%;
  font-size: 1em;
  padding: 1em;
  margin-bottom: 1em;
  border-radius: 8px;
  border: 1px solid #ccc;
}

.botoes {
  display: flex;
  gap: 1em;
}

input[type="submit"], button {
  padding: 0.7em 1.5em;
  font-weight: bold;
  cursor: pointer;
  border: none;
  border-radius: 6px;
  background-color: #4285f4;
  color: white;
  transition: 0.3s;
}

input[type="submit"]:hover, button:hover {
  background-color: #3367d6;
}

#loading {
  display: none;
  margin-top: 1em;
  font-style: italic;
}

.resposta {
  margin-top: 2em;
  padding: 1em;
  background: #e3f2fd;
  border-radius: 8px;
}

body.dark .resposta {
  background: #263238;
}
```

---

## ✅ Executar

1. Coloque sua chave da API no `.env`
2. Instale as libs:
```bash
pip install flask python-dotenv google-generativeai
```

3. Execute:
```bash
python app.py
```

4. Acesse: [http://localhost:5000](http://localhost:5000)

---

Pronto! Você tem um app Gemini com Flask, responsivo, bonito e funcional. 🚀
