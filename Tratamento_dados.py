# -*- coding: utf-8 -*-
"""
Arquivo: Tratamento_dados.py
Descrição: Faremos o download da base de dados utilizadas
Autor: João Lucas
Data de Criação: 01/06/2024
Última Atualização: 02/07/2024
"""

# Importações necessárias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Lendo dados obtidos no site da B3: "https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/series-historicas/"

# Anos que serão analisados
anos = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

# Incluindo os campos de acordo com especificações
tamanho_campos = [2,8,2,12,3,12,10,3,4,13,13,13,13,13,13,13,5,18,18,13,1,8,7,13,12,3]

base_100 = ["preco_abertura", "preco_maximo",
            "preco_minimo", "preco_medio",
            "preco_ultimo_negocio", "preco_melhor_oferta_compra", 
            "preco_melhor_oferta_venda", "volume_total_negociado",
            "preco_exercicio", "preco_exercicio_pontos"]

# Lista para armazenar os DataFrames
df_list = []

# Loop através dos anos para ler cada arquivo correspondente
for ano in anos:
    file = f'COTAHIST_A{ano}.TXT'
    
    try:
        # Leia o arquivo em um DataFrame
        df = pd.read_fwf(file, widths = tamanho_campos, header=0)
        
        df.columns = ["tipo_registro", "data_pregao",
              "cod_bdi", "cod_negociacao",
              "tipo_mercado","nome_empresa",
              "especificacao_papel", "prazo_dias_merc_termo",
              "moeda_referencia", "preco_abertura",
              "preco_maximo", "preco_minimo",
              "preco_medio", "preco_ultimo_negocio",
              "preco_melhor_oferta_compra", "preco_melhor_oferta_venda",
              "numero_negocios", "quantidade_papeis_negociados",
              "volume_total_negociado", "preco_exercicio",
              "ìndicador_correcao_precos", "data_vencimento" ,
              "fator_cotacao", "preco_exercicio_pontos",
              "codigo_isin", "num_distribuicao_papel"]
        
        # Eliminar última linha
        df = df.drop(df.index[-1])
        
        # Os preços que possuem vígula precisam ser divididos por 100
        for col in base_100:
            if col in df.columns:
                df[col] = df[col] / 100
                
        # Adicionando o DataFrame à lista
        df_list.append(df)
    except FileNotFoundError:
        print(f'O arquivo {file} não foi encontrado e será ignorado.')

# Combine todos os DataFrames em um único
df_bov = pd.concat(df_list, ignore_index=True)

df_bov['data_pregao'] = pd.to_datetime(df_bov['data_pregao'], format = '%Y%m%d')
df_bov['data_vencimento'] = pd.to_datetime(df_bov['data_vencimento'], format = '%Y%m%d', errors='coerce')

# df_bov contém todos os dados de cotações do Ibovesba de 2013 a 2023

# Selecionando as colunas com as informações das opções que utilizaremos
df_op = df_bov[["data_pregao", "tipo_mercado", "nome_empresa",
               "cod_negociacao", "preco_exercicio", "preco_ultimo_negocio",
               "data_vencimento","numero_negocios" ]]

df_op = df_op[(df_op['nome_empresa'] == "VALE")&
              #(df_op['numero_negocios']>50)&
              (df_op["tipo_mercado"] == 70)] # para inserir op de venda"|(df_op["tipo_mercado"] == 80)]"


# Formando df para o ativo objeto VALE3
df_vale = df_bov[["data_pregao", "tipo_mercado",
                  "nome_empresa", "cod_negociacao",
                  "preco_ultimo_negocio"]]

# Selecionando ações da Vale
df_vale = df_vale[df_vale['cod_negociacao'].str.contains("VALE3")&
                  (df_vale["tipo_mercado"] == 10)]

# Criando coluna com retornos logarítmicos 
# df_vale['retorno'] = np.log(df_vale['preco_ultimo_negocio']).pct_change()
df_vale['retorno'] = np.log(df_vale['preco_ultimo_negocio'] / df_vale['preco_ultimo_negocio'].shift(1))
df_vale = df_vale.dropna()

# Resetanto index
df_vale = df_vale.reset_index().drop('index', axis=1)

# Usando merge para consolidar os dois df, incluindo o preço do ativo objeto.

df_vale_merge = df_vale[['data_pregao', 'preco_ultimo_negocio']]

df_vale_merge = df_op.merge(df_vale_merge, left_on = 'data_pregao', right_on= 'data_pregao')


# ------ DataFrames Abaixo ------

# Total de dados Ibovespa
#df_bov.to_csv('df_bov.csv', index = False)

# Dados de opções de compra vale
#df_op.to_csv('df_op.csv', index = False)


# Dados das ações da vale
df_vale.to_csv('df_vale.csv', index = False)

# Dados com opções e ações da Vale consolidadas
df_vale_merge.to_csv('df_vale_merge.csv', index = False)

# -------------------------------

# A partir de df_vale, temos novo dataframe com coluna 'vol_garch' feita no R
df_vale_garch = pd.read_csv("df_vale_garch.csv")

# Ajustando merge
df_vale_garch['data_pregao'] = pd.to_datetime(df_vale_garch['data_pregao'])

df_vale_garch = df_vale_garch[['data_pregao',  'garch_vol']]

df_vale_merge = df_vale_merge.merge(df_vale_garch, left_on = 'data_pregao', right_on= 'data_pregao')

# inserindo selic diária

from bcb import sgs

selic = sgs.get({'selic':11}, start = '2015-01-01', end = '2023-12-31').reset_index()
selic.columns = ['data_pregao', 'selic']

selic['data_pregao'] = pd.to_datetime(selic['data_pregao'])

selic['selic'] = selic['selic'] / 100


# Gráfico taxa SELIC 2015 a 2023
plt.figure(figsize=(10, 6))
plt.plot(selic['data_pregao'], selic['selic'], label='SELIC', color='b')
plt.xlabel('Data')
plt.ylabel('Taxa SELIC (%)')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
# Show the plot
plt.show()


df_vale_merge = df_vale_merge.merge(selic, left_on = 'data_pregao', right_on= 'data_pregao')

# Diferenciando opções ITM ATM e OTM

# Se op_price for > 1,05 do strike = ITM
# Se op_price for < 1,05 > = 0,95 do strike = ATM
# Se op_price for < 0,95 do strike = OTM

df_vale_merge['moneyness'] = 'OTM'

df_vale_merge.loc[df_vale_merge['preco_ultimo_negocio_y'] > 1.05 * df_vale_merge['preco_exercicio'],
                  'moneyness'] = 'ITM'

df_vale_merge.loc[(df_vale_merge['preco_ultimo_negocio_y'] <= 1.05 * df_vale_merge['preco_exercicio']) & 
                  (df_vale_merge['preco_ultimo_negocio_y'] >= 0.95 * df_vale_merge['preco_exercicio']),
                  'moneyness'] = 'ATM'



#------ Data Frame consolidado Final -----

# Dados com opções, ações da Vale e volatilidade consolidados
df_vale_merge.to_csv('df_vale_merge.csv', index = False)