# -*- coding: utf-8 -*-
"""
Arquivo: Black_Scholes.py
Descrição: Aplicando Black Scholes
Autor: João Lucas
Data de Criação: 09/06/2024
Última Atualização: 23/07/2024
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

df_vale_merge = pd.read_csv('df_vale_merge.csv')
df_price = pd.read_csv('df_vale_merge.csv')

df_price['data_pregao'] = pd.to_datetime(df_price['data_pregao'], errors='coerce')
df_price['data_vencimento'] = pd.to_datetime(df_price['data_vencimento'], errors='coerce')

op_price = pd.DataFrame()
op_price['data_pregao'] = df_price['data_pregao']
op_price['data_vencimento'] = df_price['data_vencimento']
op_price['cod_negociacao'] = df_price['cod_negociacao']#+ '-' + df_price['data_pregao'].dt.year.astype(str)
#op_price['cod_negociacao'] = df_price['cod_negociacao']+ '-' + df_price['data_pregao'].dt.year.astype(str)
def criar_cod_negociacao(row):
    ano_pregao = row['data_pregao'].year
    ano_vencimento = row['data_vencimento'].year
    if ano_pregao != ano_vencimento:
        return f"{row['cod_negociacao']}-{ano_pregao}-{ano_vencimento}"
    else:
        return f"{row['cod_negociacao']}-{ano_pregao}"
# Aplicar a função para criar a nova coluna 'cod_negociacao'
op_price['cod_negociacao'] = df_price.apply(criar_cod_negociacao, axis=1)
# op_price['T'] = np.busday_count(df_price['data_pregao'].values.astype('datetime64[D]'), df_price['data_vencimento'].values.astype('datetime64[D]')) / 252
op_price['T'] = ((df_price['data_vencimento'] - df_price['data_pregao'])).dt.days / 252
op_price['S'] = df_price['preco_ultimo_negocio_y']
op_price['K'] = df_price['preco_exercicio']
op_price['r'] = ((1 + df_price['selic'])**252) - 1
op_price['sigma'] = df_price['garch_vol']
op_price['cotacao_op'] = df_price['preco_ultimo_negocio_x']
op_price['moneyness'] = df_price['moneyness']


# Considerando as variáveis abaixo
# S -> Preço atual do ativo subjacente
# K -> Preço de exercício da opção
# r -> Taxa de juros livre de risco
# sigma -> Volatilidade do ativo subjacente
# T -> Tempo até a expiração da opção (em anos)
# moneyness -> Classificação da opção


# Criando função para Black Scholes

def black_scholes_call(S, K, r, sigma, T):
    #d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * (T ** .5))
    #d2 = d1 - sigma * np.sqrt(T)
    d2 = d1 - sigma * (T ** .5)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Usando função para inserir os valores estimados no data frame
op_price['call_price'] = black_scholes_call(op_price['S'], op_price['K'],
                                            op_price['r'], op_price['sigma'],
                                            op_price['T'])

op_price.head()
op_price.dropna()
#op_price['call_price'] = op_price['call_price'].where(op_price['call_price'] >= 0, 0)

# Baixando os dados consolidados obtidos no repositório
# op_price.to_csv('op_price.csv', index = False)

# --------- Cálculo do R2 -------------

# Função para calculo do R2
def calculo_R2(df):
    # Convertendo para arrays numpy
    call_price = np.array(df['call_price'])
    cotacao_op = np.array(df['cotacao_op'])

    # Calculando a média das cotações reais
    mean_real = np.mean(cotacao_op)

    # Calculando o numerador e o denominador do R2
    SS_res = np.sum((cotacao_op - call_price) ** 2)
    SS_tot = np.sum((cotacao_op - mean_real) ** 2)

    # Calculando o R2
    R2 = 1 - (SS_res / SS_tot)

    return R2
calculo_R2(op_price)

# ------ Calculando MSE ----------
# Função para cálculo do MSE
def calculo_MSE(df):
    # Convertendo para arrays numpy
    call_price = np.array(df['call_price'])
    cotacao_op = np.array(df['cotacao_op'])

    # Calculando o MSE
    MSE = np.mean((cotacao_op - call_price) ** 2)

    return MSE


#### R2 por moneyness ####
bs_r2_moneyness = {
    'moneyness': [
        'ITM', 'ATM', 'OTM'
        ],
    'R^2': [
        calculo_R2(op_price[op_price['moneyness'] == 'ITM']),
        calculo_R2(op_price[op_price['moneyness'] == 'ATM']),
        calculo_R2(op_price[op_price['moneyness'] == 'OTM'])
    ]
      }

bs_r2_moneyness = pd.DataFrame(bs_r2_moneyness)
print(bs_r2_moneyness)


#### R2 por Ano ###
bs_r2_ano = {
    'Ano': [
        '2015','2016', '2017', '2018',
        '2019','2020', '2021', '2022', '2023'
          ],
    'R^2':   [ 
        calculo_R2(op_price[op_price['data_pregao'].dt.year == 2015]),
        calculo_R2(op_price[op_price['data_pregao'].dt.year == 2016]),
        calculo_R2(op_price[op_price['data_pregao'].dt.year == 2017]),
        calculo_R2(op_price[op_price['data_pregao'].dt.year == 2018]),
        calculo_R2(op_price[op_price['data_pregao'].dt.year == 2019]),
        calculo_R2(op_price[op_price['data_pregao'].dt.year == 2020]),
        calculo_R2(op_price[op_price['data_pregao'].dt.year == 2021]),
        calculo_R2(op_price[op_price['data_pregao'].dt.year == 2022]),
        calculo_R2(op_price[op_price['data_pregao'].dt.year == 2023])
          ]
        }       

bs_r2_ano = pd.DataFrame(bs_r2_ano)
print(bs_r2_ano)


