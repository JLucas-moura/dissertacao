# -*- coding: utf-8 -*-
"""
Arquivo: RNN_model.py
Descrição: Modelo Redes Neurais para precificação de opções
Autor: João Lucas
Data de Criação: 06/07/2024
Última Atualização: 24/07/2024
"""


import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

########################################################

df = pd.read_csv('df_vale_merge.csv')

df['data_pregao'] = pd.to_datetime(df['data_pregao'], errors='coerce')
df['data_vencimento'] = pd.to_datetime(df['data_vencimento'], errors='coerce')

lstm_df = pd.DataFrame()
lstm_df['data_pregao'] = df['data_pregao']
lstm_df['data_vencimento'] = df['data_vencimento']
lstm_df['K'] = df['preco_exercicio']
lstm_df['cod_negociacao'] = df['cod_negociacao']#+ '-' + df['data_pregao'].dt.year.astype(str)
#lstm_df['cod_negociacao'] = df['cod_negociacao']+ '-' + df['data_pregao'].dt.year.astype(str)
def criar_cod_negociacao(row):
    ano_pregao = row['data_vencimento'].year
    strike_price = row['K']  # Utilizando o valor de strike (K)
    return f"{row['cod_negociacao']}-{ano_pregao}-{strike_price}"
# Aplicar a função para criar a nova coluna 'cod_negociacao'
lstm_df['cod_negociacao'] = lstm_df.apply(criar_cod_negociacao, axis=1)
lstm_df['T'] = ((df['data_vencimento'] - df['data_pregao'])).dt.days / 252
lstm_df['S'] = df['preco_ultimo_negocio_y']
lstm_df['r'] = ((1 + df['selic'])**252) - 1
lstm_df['sigma'] = df['garch_vol']
lstm_df['cotacao_op'] = df['preco_ultimo_negocio_x']
lstm_df['moneyness'] = df['moneyness'].astype('str')

lstm_df.set_index('data_pregao', inplace=True)

#Selecionando as variáveis que serão utilizadas
nome_entrada = ['T', 'S', 'K', 'r', 'sigma', 'cotacao_op', 'moneyness']
entrada = lstm_df[nome_entrada]
cotacao_op = entrada.pop('cotacao_op')

# Definando as variáveis de entrada
inputs = {}

for name, nome_entrada in entrada.items():
    dtype = nome_entrada.dtype
    if dtype == object:
        dtype = tf.string # Criando tensor contendo strings para dados 'object'
    else:
        dtype = tf.float32 # Criando tensor contendo float 
        
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)


###### processo de normalização das entradas numéricos ######
# definindo variáveis numéricas para normalizar
numeric_inputs = {name: input for name, input in inputs.items() if input.dtype==tf.float32}

x = tf.keras.layers.Concatenate(name='concatenate_numeric')(list(numeric_inputs.values()))
norm = tf.keras.layers.Normalization()
norm.adapt(np.array(lstm_df[numeric_inputs.keys()]))
numeric_inputs_finais = norm(x)

inputs_finais = [numeric_inputs_finais]

###### Processo de one hot encoding para moneyness #######
# Criando camada StringLookup para mapear strings e transformar em inteiros
lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(entrada['moneyness']))

# Ajusta o argumento 'num_tokens' para CategoryEncoding
hot_cod = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size(), output_mode='one_hot')
    
# Aplica a transformação
x = lookup(inputs['moneyness'] )
x = hot_cod(x)
    
inputs_finais.append(x)

# Concatenar dados de entrada
concat_inputs_finais = tf.keras.layers.Concatenate(name='concatenate_all')(inputs_finais)

# Modelo para processamento das variavéis
model_process = tf.keras.Model(inputs, concat_inputs_finais)

# Gráfico resumo dos processamentos
tf.keras.utils.plot_model(model = model_process,
                          rankdir="LR", dpi=150,
                          show_shapes=True, expand_nested = True,
                          show_layer_names = True,
                          to_file='grafico_modelo.png')

####### Visualizando exemplo dos dados processados #######
# Pegando uma amostra de 5 linhas dos dados
sample_df = lstm_df.head(5)

sample_inputs = {}
for name in inputs.keys():
    sample_inputs[name] = tf.convert_to_tensor(sample_df[name].values.reshape(-1, 1))
    
# "Prevendo" o output do modelo para as amostras retiradas acima
processed_data = model_process.predict(sample_inputs)

# Visualizando os dados processados
print(processed_data)

############ Dados de Treino e Teste ############
# Treinaremos o modelo com os anos de 2015 a 2022
# O teste será com o ano de 2023

# Seleciona as colunas necessárias
dataset = lstm_df[['T', 'S', 'K', 'r', 'sigma', 'moneyness']]

# Divide em treino e teste baseado no ano de 2023, acessando 'data_pregao' pelo índice
dataset_treino = dataset[lstm_df.index.year < 2023]
dataset_teste = dataset[lstm_df.index.year >= 2023]

cotacao_op = lstm_df['cotacao_op']
cotacao_op_treino = cotacao_op[lstm_df.index.year < 2023]
cotacao_op_teste = cotacao_op[lstm_df.index.year >= 2023]

# Converte os datasets para tf.data.Dataset
ds_treino = tf.data.Dataset.from_tensor_slices((dict(dataset_treino), cotacao_op_treino))
ds_teste = tf.data.Dataset.from_tensor_slices((dict(dataset_teste), cotacao_op_teste))

# Definindo Batch size

#codigos_por_dia = lstm_df.groupby('data_pregao')['cod_negociacao'].nunique()

#batch_size = round(codigos_por_dia.mean())

batch_size = 62

ds_treino = ds_treino.batch(batch_size)
ds_teste = ds_teste.batch(batch_size)


# Visualizando as primeiras amostras
for features, label in ds_treino.take(1):
    print(f'Features: {features}')
    print(f'Label: {label}')

    
############## Criando Modelo LSTM ##############

def get_model(pre_process_model, inputs):
    body = tf.keras.Sequential([
        tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    inputs_finais = pre_process_model(inputs)
    result = body(inputs_finais)
    model = tf.keras.Model(inputs, result)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae", dtype=None)]
    )

    return model

model = get_model(model_process, inputs)


# Checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint.weights.h5',  # Arquivo com a extensão correta
    save_freq='epoch',
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

lr = tf.keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.2, 
                                          min_delta=0.001, patience=5,
                                          verbose=1 )

csv = tf.keras.callbacks.CSVLogger("results.csv")

es = tf.keras.callbacks.EarlyStopping( monitor='val_loss', min_delta=0.01, 
                                      patience=20, verbose=1 )

model.fit(
    ds_treino,
    validation_data = ds_teste,
    epochs = 200,
    callbacks = [checkpoint, lr, csv, es]
    )

# Resultados do modelo
resultados = pd.read_csv('results.csv')
resultados.head()

model.save('LSTM_model.h5')


############## Gráfico dos resultados #################

# Plotando as métricas de Loss e MAE (Mean Absolute Error) para treino e validação
fig, axes = plt.subplots(2, 1, figsize=(8, 9))

# Gráfico de Loss
axes[0].plot(resultados['epoch'], resultados['loss'], label='Train Loss', color='blue')
axes[0].plot(resultados['epoch'], resultados['val_loss'], label='Validation Loss', color='orange')
axes[0].set_title('Loss over Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Gráfico de MAE
axes[1].plot(resultados['epoch'], resultados['mae'], label='Train MAE', color='green')
axes[1].plot(resultados['epoch'], resultados['val_mae'], label='Validation MAE', color='red')
axes[1].set_title('Mean Absolute Error (MAE) over Epochs')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('MAE')
axes[1].legend()

plt.tight_layout()

# Salvando os resultados do modelo
fig.savefig('LSTM_resultados.png')

plt.show()


# Carregando o modelo desejado salvo
#model = load_model('LSTM_model.h5')

# Obtenha as previsões para o dataset de teste
predicoes = model.predict(ds_teste)

# Convertendo o dataset de teste em um DataFrame para associar as previsões
dataset_teste_df = lstm_df[lstm_df.index.year >= 2023].copy()


########## funções para avaliação do resultado ##########
# Função de Black Scholes
def black_scholes_call(S, K, r, sigma, T):
    #d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * (T ** .5))
    #d2 = d1 - sigma * np.sqrt(T)
    d2 = d1 - sigma * (T ** .5)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price
# Função do R2
def calculo_R2(y, y_pred):
    # Convertendo para arrays numpy
    call_price = np.array(y_pred)
    cotacao_op = np.array(y)

    # Calculando a média das cotações reais
    mean_real = np.mean(cotacao_op)

    # Calculando o numerador e o denominador do R2
    SS_res = np.sum((cotacao_op - call_price) ** 2)
    SS_tot = np.sum((cotacao_op - mean_real) ** 2)

    # Calculando o R2
    R2 = 1 - (SS_res / SS_tot)

    return R2
# Função MSE
def calculo_MSE(y, y_pred):
    # Convertendo para arrays numpy
    call_price = np.array(y_pred)
    cotacao_op = np.array(y)

    # Calculando o MSE
    MSE = np.mean((cotacao_op - call_price) ** 2)
    MSE = MSE**(0.5)

    return MSE


dataset_teste_df['call_price_bs'] = black_scholes_call(dataset_teste_df['S'], dataset_teste_df['K'],
                                            dataset_teste_df['r'], dataset_teste_df['sigma'],
                                            dataset_teste_df['T'])


# Com as previsões estejam alinhadas as amostras do dataset de teste,
# adicionando as previsões como uma nova coluna no DataFrame
dataset_teste_df['call_price_lstm'] = predicoes

# Exibir as primeiras linhas para verificar
dataset_teste_df.head()

dataset_teste_df.to_csv('dataset_teste_df.csv', index = False)


########   Comparação de desempenho  ########

# Calculo do R^2
R2_bs = calculo_R2(dataset_teste_df['cotacao_op'], dataset_teste_df['call_price_bs'])
R2_lstm = calculo_R2(dataset_teste_df['cotacao_op'], dataset_teste_df['call_price_lstm'])

# Calculo do MSE
MSE_bs = calculo_MSE(dataset_teste_df['cotacao_op'], dataset_teste_df['call_price_bs'])
MSE_lstm = calculo_MSE(dataset_teste_df['cotacao_op'], dataset_teste_df['call_price_lstm'])

# R2 geral
print(R2_bs, R2_lstm)
# MSE geral
print(MSE_bs, MSE_lstm)


# Vendo a performance dos dados por Moneyness:

############ R^2 ###################
# Lista para armazenar os resultados
resultados = []

# Calcular R^2 para cada categoria de moneyness
for categoria in ['ITM', 'ATM', 'OTM']:
    # Filtrar o DataFrame por categoria
    df_categoria = dataset_teste_df[dataset_teste_df['moneyness'] == categoria]
    
    # Calcular R^2 lstm
    r2_lstm = calculo_R2(df_categoria['cotacao_op'], df_categoria['call_price_lstm'])
    r2_bs = calculo_R2(df_categoria['cotacao_op'], df_categoria['call_price_bs'])
    # Armazenar o resultado lstm
    resultados.append({'moneyness': categoria,'Black Scholes':r2_bs ,'LSTM': r2_lstm})
    
# Criar um DataFrame com os resultados
df_resultados_r2 = pd.DataFrame(resultados)

print(df_resultados_r2)

############# MSE #################
resultados = []
for categoria in ['ITM', 'ATM', 'OTM']:
    # Filtrar o DataFrame por categoria
    df_categoria = dataset_teste_df[dataset_teste_df['moneyness'] == categoria]
    
    # Calcular R^2 lstm
    MSE_lstm = calculo_MSE(df_categoria['cotacao_op'], df_categoria['call_price_lstm'])
    MSE_bs = calculo_MSE(df_categoria['cotacao_op'], df_categoria['call_price_bs'])
    # Armazenar o resultado lstm
    resultados.append({'moneyness': categoria,'Black Scholes':MSE_bs ,'LSTM': MSE_lstm})
    
# Criar um DataFrame com os resultados
df_resultados_MSE = pd.DataFrame(resultados)

print(df_resultados_MSE)

######## desempenho por ano ##############
resultados = []

# Filtrar os dados por ano, 2022 e 2023, considerando o index 'data_pregao'
#for ano in [2022, 2023]:
    # Filtrar o DataFrame pelo ano
    #df_ano = dataset_teste_df[dataset_teste_df.index.year == ano]
    
    # Agora calcular as métricas de MSE por categoria de moneyness para o ano
    #for categoria in ['ITM', 'ATM', 'OTM']:
        # Filtrar o DataFrame pela categoria de moneyness
    #    df_categoria = df_ano[df_ano['moneyness'] == categoria]
        
        # Calcular MSE para LSTM e Black-Scholes
    #    MSE_lstm = calculo_R2(df_categoria['cotacao_op'], df_categoria['call_price_lstm'])
    #    MSE_bs = calculo_R2(df_categoria['cotacao_op'], df_categoria['call_price_bs'])
        
        # Armazenar o resultado para o ano e categoria atual
    #    resultados.append({
    #        'ano': ano,
    #        'moneyness': categoria,
    #        'Black Scholes': MSE_bs,
    #        'LSTM': MSE_lstm
    #    })
# Criar um DataFrame com os resultados
#df_resultados_ano = pd.DataFrame(resultados)
# Exibir os resultados
#print(df_resultados_ano)


############ Plot de algumas opções ###############
# ITM -> VALED768-2023-73.3
# ATM -> VALEA117-2023-96.16
# OTM -> VALED925-2023-92.3
codigo_opcao = 'VALED768-2023-73.3'
df_opcao = dataset_teste_df[dataset_teste_df['cod_negociacao'] == codigo_opcao]

# Certifique-se de que o índice é do tipo datetime
df_opcao.index = pd.to_datetime(df_opcao.index)

# Ordenando o DataFrame pelo índice
df_opcao = df_opcao.sort_index()

# Criando subplots (duas linhas)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

# Plot da primeira linha (cotação e preços de opções)
axes[0].plot(df_opcao.index, df_opcao['cotacao_op'], label='Preço Observado', color='blue')
axes[0].plot(df_opcao.index, df_opcao['call_price_bs'], label='Black-Scholes', color='orange')
axes[0].plot(df_opcao.index, df_opcao['call_price_lstm'], label='LSTM', color='red')

axes[0].set_title('Cotação opção vs valores estimados')
axes[0].set_ylabel('Preço')
axes[0].legend()
axes[0].grid()

# Plot da segunda linha (preço do ativo subjacente e strike)
axes[1].plot(df_opcao.index, df_opcao['S'], label='Ativo objeto', color='black')
axes[1].plot(df_opcao.index, df_opcao['K'], label='Preço de exercício', color='green')
axes[1].set_title('Ativo objeto vs preço de exercício')
axes[1].set_xlabel('Data')
axes[1].set_ylabel('Preço')
axes[1].legend()
axes[1].grid()

# Ajustando o layout
plt.tight_layout()

# Exibindo o gráfico
plt.show()
