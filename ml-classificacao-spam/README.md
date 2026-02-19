# ✅ **Aprendizagem por Classificação — Detecção de Spam**
---
## 🎯 **Problema** - Classificar mensagens de texto como:
- 0 → Não Spam
- 1 → Spam

Trata-se de um problema clássico de Aprendizagem Supervisionada, mais especificamente de Classificação Binária, amplamente utilizado em aplicações reais como filtros de e-mail, mensagens SMS, chatbots e sistemas antifraude.

---

## 🎓 Objetivo do Laboratório
**Este projeto demonstra**:
- Evolução incremental de ML
- Separação de responsabilidades
- Estrutura profissional
- Fundamentação matemática
- Organização modular
- Avaliação estatística robusta

---

## 🏗 **Estrutura do Projeto**

**ml-classificacao-spam**/\
│\
├── classificacao_spam_simple.py\
├── classificacao_spam.py\
├── gerar_dataset.py\
└── README.md\

---

## 🧠 **Modelo Utilizado**

- Regressão Logística
- Aplicada sobre vetores numéricos derivados de texto (NLP)
- Implementação com scikit-learn

---


## 💻 **Implementações**

### 🔹 `classificacao_spam_simple.py`

Esta versão implementa uma abordagem direta e objetiva para classificação de spam.

#### Principais características:

- Dataset simples e criado manualmente
- Vetorização básica com **`CountVectorizer`**
- Separação treino/teste
- Modelo de Regressão Logística
- Avaliação com métricas padrão (simples)

**Objetivo**: demonstrar pipeline mínimo funcional.

#### Fluxo:

- Criação do dataset
- Vetorização do texto
- Treinamento do modelo
- Avaliação com:
- Precisão
- Recall
- F1-score
- Acurácia

Essa abordagem demonstra o pipeline mínimo funcional para um problema de NLP.

### 🔹 `classificacao_spam.py`

Esta versão evolui estruturalmente o projeto, tornando-o mais próximo de um padrão profissional utilizado em produção. Ela introduz melhorias arquiteturais, estatísticas e organizacionais.

Versão estruturada com padrão mais próximo de produção.

#### 🔹 1. Classe GerarDataset

**Arquivo relacionado**: gerar_dataset.py

Foi criada uma classe responsável por gerar dinamicamente datasets sintéticos de mensagens classificadas como spam (1) ou não spam (0).

**Características**:
- Método @staticmethod gerar_dataset
- Permite gerar dataset sem instanciar a classe

**Controle de**:
- Tamanho do dataset
- Proporção de spam (ex: 55%)

**Inclusão de**:
- Frases realistas
- Frases ambíguas
- Ruído linguístico (erros de digitação e variações)

**Benefício:**

O problema deixa de ser trivial e se aproxima mais de cenários reais de NLP, tornando o modelo mais robusto.

#### 🔹 2. Estruturação com Pipeline

**Foi adotado o Pipeline do scikit-learn para encadear**:
- Vetorização com TF-IDF
- Classificação com Regressão Logística

**Benefícios**:
- Garante aplicação consistente no treino e teste
- Evita data leakage
- Melhora organização do código
- Aproxima a solução de padrões produtivos

#### 🔹 3. Melhoria na Vetorização

**Substituição de**:

- _CountVectorizer_ → _TfidfVectorizer_
- Melhorias implementadas:
- Uso de ngram_range=(1,2) (unigrams + bigrams)
- Melhor ponderação da importância das palavras e termos
- Redução do impacto de termos muito frequentes
- Maior robustez contra ruído linguístico

#### 🔹 4. Configuração do Modelo

- _class_weight='balanced'_ para ajustar os pesos das classes para lidar com desbalanceamento
- Prepara o modelo para possíveis desequilíbrios
- _max_iter=1000_ para adequação do modelo em datasets diversos
- Garante convergência em datasets maiores

**Separação explícita**:
- **X** → Variáveis independentes
- **y** → Variável alvo

#### 🔹 5. Estratégia de Avaliação

📌 **Divisão Treino/Teste**

- Uso de **`train_test_split`** para divisão dos dados em treino e teste
- **`random_state`** para garantia de reprodutibilidade dos resultados
- Possibilidade de uso de **`stratify=y`** para manter proporção das classes

📌 **Relatório de Classificação**

**Exibição de métricas**:
- Precisão
- Recall
- F1-score
- Acurácia
- Permite análise detalhada por classe:
- Spam
- Não Spam

#### 🔹 6. Validação Cruzada (Cross-Validation)

**Aplicação de**: 
- _cross_val_score_
- Utilizando a Pipeline completa.

**Benefícios**:
- Vetorização recalculada dentro de cada _fold_
- Evita vazamento de dados
- Estimativa mais robusta da generalização
- Acurácia média como indicador de estabilidade do modelo

---

## 🚀 Como Executar

### 1️⃣ Criar Ambiente Virtual

```bash
python -m venv venv
```
**Ativar Windows**:

```bash
venv\Scripts\activate
```

**Ativar Linux / Mac**:

```bash
source venv/bin/activate
```

### 2️⃣ Executar

**Versão simples**:

```bash
py classificacao_spam_simple.py
```

**Versão estruturada**:

```bash
py classificacao_spam.py
```


## 📊 Métricas Avaliadas
- Precisão
- Recall
- F1-Score
- Acurácia
- Validação Cruzada

## 🔬 Conceitos Demonstrados
- Aprendizagem Supervisionada
- Classificação Binária
- NLP
- TF-IDF
- Engenharia de Features
- Balanceamento de Classes
- Pipeline do Scikit-Learn
- Cross-Validation
- Prevenção de Data Leakage

## 🧩 Evoluções Futuras
- GridSearchCV
- Matriz de confusão
- Curva ROC
- Word Embeddings
- Deploy com FastAPI
- Dockerização
- Integração com CI/CD

## 🤝 Contribuição

**Contribuições são bem-vindas!**

**Passos**:
- git checkout -b feature/nova-melhoria
- git commit -m "feat: nova melhoria"
- git push origin feature/nova-melhoria

**Abra um Pull Request** 🚀

## 👨‍💻 **Autor**

_George Mendonça_

_AI • Data • Machine Learning • GenAI • Data Architecture • Data Governance • DataOps_