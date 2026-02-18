'''
✅ Aprendizagem por Classificação — Detecção de Spam

🎯 Problema - Classificar mensagens como:
    0 = Não spam
    1 = Spam

🧠 Modelo: Regressão Logística

💻 Implementação Python

@author: @George Mendonça
@date: 2026-02-18
'''

# Importando bibliotecas
from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

# Carregando o dataset
dataset = {
    'mensagem': [
        'Oferta exclusiva para você',
        'Reunião confirmada para amanhã',
        'Ganhe prêmios agora',
        'Segue o relatório financeiro',
        'Promoção válida até hoje',
        'Vamos almoçar amanhã?',
        'Você foi selecionado',
        'Atualização do projeto enviada',
        'Desconto imperdível',
        'Confirmando presença na reunião'        
        ],
    'spam': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

# Criando DataFrame
df = pd.DataFrame(dataset)

''' Pré-processamento dos dados - Conveter texto em números (Bag of Words) '''
vetorizador = CountVectorizer() # Criando o vetorizador de texto
X = vetorizador.fit_transform(df['mensagem']) # Transformando as mensagens em uma matriz de contagem
y = df['spam'] # Variável alvo

''' Divisão dos dados em treino e teste '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Dividindo os dados em treino e teste (70% treino, 30% teste)

''' Treinando o modelo de Regressão Logística '''
modelo = LogisticRegression() # Instanciando o modelo
modelo.fit(X_train, y_train) # Treinando o modelo

''' Avaliação do modelo '''
previsoes = modelo.predict(X_test) # Fazendo previsões

''' Exibindo relatório de classificação '''
relatorio = classification_report(y_test, previsoes) # Avaliação do modelo usando o relatório de classificação
print("Relatório de Classificação:") # Imprimindo o título do relatório

''' Imprimindo o relatório de classificação
    Que inclui métricas como precisão, recall e f1-score para cada classe (spam e não spam) '''
print(relatorio)