'''
Docstring for ml-classificacao-span.gerar-dataset

Geração de um dataset de mensagens para classificação de spam.

Este módulo cria um conjunto de dados fictício contendo mensagens de texto e suas respectivas classificações como spam ou não spam.

O dataset gerado pode ser utilizado para treinar e avaliar modelos de aprendizado de máquina para a detecção de spam em mensagens de texto.

@author: @George Mendonça
@date: 2026-02-18

'''

import pandas as pd
import random
import os

''' GerarDataset é uma classe responsável por criar um dataset sintético de mensagens de texto, classificando-as como spam ou não spam. '''
class GerarDataset:

    ''' O método gerar_dataset é um método estático que gera um dataset de mensagens de texto, onde cada mensagem é classificada como spam (1) ou não spam (0).
        Ele recebe dois parâmetros opcionais: tamanho, que define o número total de mensagens a serem geradas (padrão é 500), e porcentagem_spam,
        que define a proporção de mensagens que devem ser classificadas como spam (padrão é 0.55, ou seja, 55% das mensagens serão spam).    
    '''
    @staticmethod
    def gerar_dataset(tamanho=500, porcentagem_spam=0.55):
        """ Gerar um dataset de mensagens de texto com classificação de spam.
        :param tamanho: O número total de mensagens a serem geradas (padrão: 500)
        :param porcentagem_spam: A proporção de mensagens que devem ser classificadas como spam (padrão: 0.55), implicitamente não ficará com 45% de mensagens não spam.
        :return: Um DataFrame do pandas contendo as mensagens e suas classificações de spam.        
        """

        # Frases de exemplo para mensagens de spam e não spam
        frases_spam = [
            'Oferta exclusiva para você',
            'Ganhe prêmios agora',
            'Promoção válida até hoje',
            'Você foi selecionado',
            'Desconto imperdível',
            'Ganhe dinheiro rápido',
            'Promoção exclusiva',
            'Clique agora',
            'Você ganhou um prêmio',
            'Oferta imperdível',
            'Desconto só hoje',
            'Parabéns você foi selecionado',
            'Acesse o link agora',
            'Última chance',
            'Oferta limitada',
            'Multiplique sua renda',
            'Crédito aprovado imediatamente',
            'Investimento garantido',
            'Lucro rápido e fácil',
            'Promoção exclusiva para reunião VIP',
            'Projeto premiado, clique agora',
            'Atualização de segurança. Clique aqui.', # Frase ambígua que pode ser interpretada como spam ou não spam, dependendo do contexto
            'G4nh3 dinh3iro rápid0', # Ruído para simular mensagens de spam com erros de digitação
            'Promoçãooo exclusiva!!!', # Ruído para simular mensagens de spam com erros de digitação
            'Cliqueee aquiiii' # Ruído para simular mensagens de spam com erros de digitação
        ]
        frases_nao_spam = [
            'Reunião confirmada para amanhã',
            'Segue o relatório financeiro',
            'Vamos almoçar amanhã?',
            'Atualização do projeto enviada',
            'Confirmando presença na reunião',
            'Boa tarde, tudo bem?',
            'Segue anexo o documento',
            'Precisamos conversar',
            'Enviei o e-mail',
            'Projeto aprovado',
            'Relatório final disponível',
            'Reunião reagendada',
            'Pagamento confirmado',
            'Atualização contratual enviada',
            'Reunião cancelada às 15h',
            'Segue o resumo da reunião',
            'Vamos marcar uma reunião para discutir o projeto',
            'Atualização da agenda enviada',
            'Confirmando presença com a equipe para a apresentação de terça-feira',
            'Reunião sobre campanha promocional',
            'Projeto de marketing com desconto',
            'Atualização de segurança do sistema interno', # Frase ambígua que pode ser interpretada como spam ou não spam, dependendo do contexto
            'Reuniã amanhã', # Ruído para simular mensagens não spam com erros de digitação
            'Projto aprovado' # Ruído para simular mensagens não spam com erros de digitação
        ]

        # Gerar mensagens de spam e não spam com base na proporção desejada e embaralhar os dados para criar um dataset equilibrado
        dados = []

        # Calculando a quantidade de mensagens de spam e não spam com base na proporção desejada (55% spam, 45% não spam, por exemplo)
        qtd_spam = int(tamanho * porcentagem_spam)
        qtd_nao_spam = tamanho - qtd_spam

        for _ in range(qtd_spam):
            dados.append([random.choice(frases_spam), 1])

        for _ in range(qtd_nao_spam):
            dados.append([random.choice(frases_nao_spam), 0])

        random.shuffle(dados)

        df = pd.DataFrame(dados, columns=["mensagem", "spam"])

        return df


# Executa apenas se rodar direto no terminal
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = GerarDataset.gerar_dataset(500)
    df.to_csv("data/mensagens_spam.csv", index=False)
    print("✅ Dataset salvo com sucesso!")