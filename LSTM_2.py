# -*- coding: utf-8 -*-
"""
Arquivo: RNN_model.py
Descrição: Modelo Redes Neurais para precificação de opções, tentativa de
           padding, para que todas as opções tenham o mesmo número de cotações
Autor: João Lucas
Data de Criação: 06/07/2024
Última Atualização: 20/07/2024
"""


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.utils import plot_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

########################################################

df = pd.read_csv('df_vale_merge.csv')

df['data_pregao'] = pd.to_datetime(df['data_pregao'], errors='coerce')
df['data_vencimento'] = pd.to_datetime(df['data_vencimento'], errors='coerce')

lstm_df = pd.DataFrame()
lstm_df['data_pregao'] = df['data_pregao']
lstm_df['data_vencimento'] = df['data_vencimento']

lstm_df['cod_negociacao'] = df['cod_negociacao']
#lstm_df['cod_negociacao'] = df['cod_negociacao']+ '-' + df['data_pregao'].dt.year.astype(str)
def criar_cod_negociacao(row):
    ano_pregao = row['data_pregao'].year
    ano_vencimento = row['data_vencimento'].year
    if ano_pregao != ano_vencimento:
        return f"{row['cod_negociacao']}-{ano_pregao}-{ano_vencimento}"
    else:
        return f"{row['cod_negociacao']}-{ano_pregao}"
# Aplicar a função para criar a nova coluna 'cod_negociacao'
lstm_df['cod_negociacao'] = lstm_df.apply(criar_cod_negociacao, axis=1)

lstm_df['T'] = ((df['data_vencimento'] - df['data_pregao'])).dt.days / 252
lstm_df['S'] = df['preco_ultimo_negocio_y']
lstm_df['K'] = df['preco_exercicio']
lstm_df['r'] = ((1 + df['selic'])**252) - 1
lstm_df['sigma'] = df['garch_vol']
lstm_df['cotacao_op'] = df['preco_ultimo_negocio_x']
lstm_df['moneyness'] = df['moneyness'].astype('str')


########################################################


# Obtenha todas as datas e todos os códigos de negociação únicos
all_dates = lstm_df['data_pregao'].unique()
all_codes = lstm_df['cod_negociacao'].unique()

# Obtenha o DataFrame de data de vencimento único por código de negociação
vencimento_por_codigo = lstm_df[['cod_negociacao', 'data_vencimento']].drop_duplicates()

# Crie um DataFrame com todas as combinações de datas e códigos
full_combinations = pd.MultiIndex.from_product([all_dates, all_codes],
                                               names=['data_pregao',
                                                      'cod_negociacao']).to_frame(index=False)

# Adicionando a data de vencimento de cada código de negociação ao DataFrame de combinações
full_combinations = pd.merge(full_combinations, vencimento_por_codigo, on='cod_negociacao', how='left')

# Merge para concatenar com o DataFrame original
df_full = pd.merge(full_combinations, lstm_df, on=['data_pregao',
                                              'cod_negociacao',
                                              'data_vencimento'], how='left')

# Preenchendo os valores ausentes com zero
df_full.fillna(0, inplace=True)

df_full['moneyness'] = df_full['moneyness'].replace(0, 'OTM')

#Deixando o datafarme em ordem de data e codigo de negociação
df_full.sort_values(by=['cod_negociacao', 'data_pregao'], inplace=True).reset_index

# Resetando o index
df_full.reset_index(drop=True, inplace=True)


#Selecionando as variáveis que serão utilizadas
nome_entrada = ['T', 'S', 'K', 'r', 'sigma', 'cotacao_op', 'moneyness']
entrada = df_full[nome_entrada]
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
norm.adapt(np.array(df_full[numeric_inputs.keys()]))
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
sample_df = df_full.head(5)

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
dataset = df_full[['data_pregao', 'T', 'S', 'K', 'r', 'sigma', 'moneyness']]

# Divide em treino e teste baseado no ano de 2023
dataset_treino = dataset[dataset['data_pregao'].dt.year < 2023]
dataset_teste = dataset[dataset['data_pregao'].dt.year >= 2023]

# Remove a coluna 'data_pregao'
dataset_treino = dataset_treino.drop(columns=['data_pregao'])
dataset_teste = dataset_teste.drop(columns=['data_pregao'])

# Filtra cotacao_op para treino e teste
cotacao_op = df_full['cotacao_op']
cotacao_op_treino = cotacao_op[df_full['data_pregao'].dt.year < 2023]
cotacao_op_teste = cotacao_op[df_full['data_pregao'].dt.year >= 2023]

# Converte os datasets para tf.data.Dataset
ds_treino = tf.data.Dataset.from_tensor_slices((dict(dataset_treino), cotacao_op_treino))
ds_teste = tf.data.Dataset.from_tensor_slices((dict(dataset_teste), cotacao_op_teste))

# Definindo Batch size

#codigos_por_dia = df_full.groupby('data_pregao')['cod_negociacao'].nunique()

#batch_size = round(codigos_por_dia.mean())

batch_size = 2108 # número de datas que temos

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
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
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
    epochs = 5,
    callbacks = [checkpoint, lr, csv, es]
    )

# Resultados do modelo
resultados = pd.read_csv('results.csv')
resultados.head()

# Salvando o modelo treinado e seus resultados
model.save('modelo_lstm_2.h5')
resultados.to_csv("resultados_modelo_2.csv")




# Gráfico dos resultados

# Plotando as métricas de Loss e MAE (Mean Absolute Error) para treino e validação
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Gráfico de Loss
axes[0].plot(resultados['epoch'], resultados['loss'], label='Train Loss', color='blue')
axes[0].plot(resultados['epoch'], resultados['val_loss'], label='Validation Loss', color='orange')
axes[0].set_title('Loss over Epochs')
axes[0].set_xlabel('Epochs')
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
plt.show()



# Carregando o modelo desejado salvo
#model_carregado = tf.keras.models.load_model('modelo_lstm_1.h5')

# Obtenha as previsões para o dataset de teste
predicoes = model.predict(ds_teste)

# Convertendo o dataset de teste em um DataFrame para associar as previsões
dataset_teste_df = df_full[df_full.index.year >= 2023].copy()

# Supondo que as previsões estejam alinhadas com as amostras do dataset de teste,
# adicione as previsões como uma nova coluna no DataFrame
dataset_teste_df['call_price'] = predicoes

# Exibir as primeiras linhas para verificar
print(dataset_teste_df.head())

dataset_teste_df



###########################################

