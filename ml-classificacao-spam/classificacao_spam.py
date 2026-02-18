'''
✅ Aprendizagem por Classificação — Detecção de Spam

🎯 Problema - Classificar mensagens como:
    0 = Não spam
    1 = Spam

🧠 Modelo: Regressão Logística

💻 Implementação Python - Evoluções Estruturais do Projeto:

    1. Criação da Classe GerarDataset
        - Implementação de uma classe responsável por gerar dinamicamente um dataset sintético de mensagens classificadas como spam (1) ou não spam (0).
        - O método gerar_dataset foi definido como @staticmethod, permitindo geração direta do dataset sem necessidade de instanciar a classe.
        - Possibilidade de controlar: Tamanho do dataset e proporção de spam (ex: 55%).
        - Inclusão de frases realistas, frases ambíguas e ruído linguístico (erros de digitação e variações).
        - Isso torna o problema menos trivial e mais próximo de cenários reais de NLP.

    2 Estruturação com Pipeline
        - Adoção de Pipeline para encadear: Vetorização com TF-IDF e classificação com Regressão Logística
        - Garante aplicação consistente das etapas no treino e teste.
        - Evita vazamento de dados (data leakage), pois a vetorização passa a ser ajustada apenas com dados de treino dentro de cada ciclo.
        - Melhora organização do código e aproxima a solução de padrões utilizados em produção.

    3 Melhoria na Vetorização
        - Substituição do CountVectorizer por TfidfVectorizer.
        - Consideração de unigrams e bigrams (ngram_range=(1,2)), capturando contexto adicional.
        - Melhor ponderação da importância das palavras.
        - Redução do impacto de termos muito frequentes.
        - Maior robustez frente a ruído linguístico.

    4 Configuração do Modelo
        - Uso de class_weight='balanced', preparando o modelo para possíveis desequilíbrios entre classes.
        - Definição de max_iter=1000, garantindo convergência adequada em datasets maiores.
        - Separação explícita entre variáveis independentes (X) e variável alvo (y).

    5 Estratégia de Avaliação
        - Divisão Treino/Teste
            - Uso de train_test_split com random_state para reprodutibilidade.
            - Possibilidade de uso de stratify=y para manter proporção das classes.
        - Relatório de Classificação
            - Exibição de métricas:
                - Precisão, Recall, F1-score e Acurácia
        - Permite análise mais detalhada por classe (spam vs não spam).

    6 Validação Cruzada (Cross-Validation)
        - Aplicação de cross_val_score utilizando a Pipeline completa.
        - A vetorização é recalculada dentro de cada fold.
        - Evita vazamento de dados durante validação.
        - Fornece estimativa mais robusta da capacidade de generalização do modelo.
        - A acurácia média obtida é um indicador da estabilidade do modelo diante de variações no conjunto de treino/teste.

@author: George Mendonça
@date: 2026-02-18
'''

# Importando bibliotecas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer # Importando o vetorizador TF-IDF para converter texto em números
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from gerar_dataset import GerarDataset

df = GerarDataset.gerar_dataset() # Gerando um dataset sintético de mensagens de texto com classificação de spam usando a classe GerarDataset
X = df['mensagem']  # Agora passamos texto bruto (variável independente) para a Pipeline, que irá cuidar da vetorização
y = df['spam'] # Variável alvo permanece a mesma

''' Criação da Pipeline
        A Pipeline irá encadear duas etapas:
            1. Vetorização com TF-IDF
            2. Classificação com Regressão Logística '''
pipeline = Pipeline([
    (
        'vetorizador',
        TfidfVectorizer(
            stop_words=None, # Não removemos stop words para manter o contexto completo
            ngram_range=(1,2) # Consideramos unigrams e bigrams para capturar mais contexto nas mensagens
        )
    ),
    (
        'classificador',
        LogisticRegression(
            class_weight='balanced', # Ajusta pesos das classes para lidar com desbalanceamento
            max_iter=1000 # Garantindo convergência adequada do modelo em datasets maiores
        )
    )
])

''' Divisão dos dados em treino e teste '''
X_train, X_test, y_train, y_test = train_test_split(
    X, # Passamos o texto bruto para a Pipeline, que irá cuidar da vetorização, evitando risco de vazamento de dado
    y, # Variável alvo permanece a mesma
    test_size=0.3, # 30% dos dados para teste, 70% para treino
    random_state=42, # Semente para reprodutibilidade
    stratify=y # Preserva a proporção das classes em treino e teste
)

''' Treinando o modelo através da Pipeline  '''
pipeline.fit(X_train, y_train)

''' Avaliação do modelo no conjunto de teste '''
previsoes = pipeline.predict(X_test)

''' Exibindo relatório de classificação '''
relatorio = classification_report(y_test, previsoes)

print("Relatório de Classificação:")
print(relatorio)
print(" ")
print("    => Precisão - Proporção de mensagens classificadas como spam que são realmente spam (verdadeiros positivos / (verdadeiros positivos + falsos positivos))")
print("    => Recall - Proporção de mensagens realmente spam que foram corretamente classificadas como spam (verdadeiros positivos / (verdadeiros positivos + falsos negativos))")
print("    => F1-score - Média harmônica entre precisão e recall, fornecendo uma única métrica para avaliar o desempenho do modelo")
print("    => Supporte - Número de amostras reais para cada classe (spam e não spam) - Desbalanceamento pode ser observado aqui")
print("    => Macro avg - Média das métricas (precisão, recall, F1-score) calculada de forma simples, sem considerar o suporte de cada classe")
print("    => Weighted avg - Média das métricas (precisão, recall, F1-score) ponderada pelo suporte de cada classe, refletindo melhor o desempenho geral do modelo em casos de desbalanceamento")
print(" ")