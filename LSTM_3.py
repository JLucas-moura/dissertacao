# -*- coding: utf-8 -*-
"""
Arquivo: LSTM_3.py
Descrição: Modelo Redes Neurais para precificação de opções com Batch Variavel.
Autor: João Lucas
Data de Criação: 06/07/2024
Última Atualização: 29/07/2024
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.utils import plot_model
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv 
import seaborn

df = pd.read_csv('df_vale_merge.csv')

df['data_pregao'] = pd.to_datetime(df['data_pregao'], errors='coerce')
df['data_vencimento'] = pd.to_datetime(df['data_vencimento'], errors='coerce')

lstm_df = pd.DataFrame()
lstm_df['data_pregao'] = df['data_pregao']
lstm_df['data_vencimento'] = df['data_vencimento']
lstm_df['K'] = df['preco_exercicio']

# Se faz necessário concatenar o cod_negociacao com o preço de exercicio
# e data de vencimento, para garantir que cada opção lançada é única, sem
# se confundir com outras opções de código similar de anos diferentes
lstm_df['cod_negociacao'] = df['cod_negociacao']

def criar_cod_negociacao(row):
    ano_pregao = row['data_vencimento'].year
    strike_price = row['K']  # Utilizando o valor de strike (K)
    return f"{row['cod_negociacao']}-{ano_pregao}-{strike_price}"

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


#---------- processo de normalização das entradas numéricos ----------
# definindo variáveis numéricas para normalizar
numeric_inputs = {name: input for name, input in inputs.items() if input.dtype==tf.float32}

x = tf.keras.layers.Concatenate(name='concatenate_numeric')(list(numeric_inputs.values()))
norm = tf.keras.layers.Normalization()
norm.adapt(np.array(lstm_df[numeric_inputs.keys()]))
numeric_inputs_finais = norm(x)

inputs_finais = [numeric_inputs_finais]

#----------- One hot encoding para moneyness -------------
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

#----------- Visualizando exemplo dos dados processados -----------
# Pegando uma amostra de 5 linhas dos dados
sample_df = lstm_df.head(5)

sample_inputs = {}
for name in inputs.keys():
    sample_inputs[name] = tf.convert_to_tensor(sample_df[name].values.reshape(-1, 1))
    
# "Prevendo" o output do modelo para as amostras retiradas acima
processed_data = model_process.predict(sample_inputs)

# Visualizando os dados processados
print(processed_data)

#---------- Dados de Treino e Teste ----------
# Treinaremos o modelo com os anos de 2015 a 2022
# O teste será com o ano de 2023

# Seleciona as colunas necessárias
dataset = lstm_df[['T', 'S', 'K', 'r', 'sigma', 'moneyness']]

# Divide em treino e teste baseado no ano de 2023, acessando 'data_pregao' pelo índice
dataset_treino = dataset[lstm_df.index.year < 2023]
dataset_teste = dataset[lstm_df.index.year >= 2023]


# Separação dos dados de treino e teste
cotacao_op = lstm_df['cotacao_op']
cotacao_op_treino = cotacao_op[lstm_df.index.year < 2023]
cotacao_op_teste = cotacao_op[lstm_df.index.year >= 2023]

# Criação dos datasets a partir dos dados de treino e teste
ds_treino = tf.data.Dataset.from_tensor_slices((dict(dataset_treino), cotacao_op_treino))
ds_teste = tf.data.Dataset.from_tensor_slices((dict(dataset_teste), cotacao_op_teste))

# Obter tamanhos de batch para cada step
batch_sizes = lstm_df[lstm_df.index.year < 2023].groupby('data_pregao')['cod_negociacao'].count()
batch_sizes = batch_sizes.tolist()

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


def custom_training_loop_per_step(model, ds_treino, ds_teste, batch_sizes, epochs, model_process, log_file='training_log.csv'):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    metric_mae = tf.keras.metrics.MeanAbsoluteError(name="mae")

    # Armazenar valores de perda e MAE para treino e validação
    loss_history = []
    mae_history = []
    val_loss_history = []
    val_mae_history = []

    # Extrai as features e targets do dataset de treino para criar batches manuais
    train_data = list(ds_treino)

    # Criar o arquivo CSV e escrever o cabeçalho
    with open(log_file, mode='w', newline='') as csvfile:
        fieldnames = ['epoch', 'step', 'loss', 'mae', 'val_loss', 'val_mae']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Resetando as métricas a cada época
            metric_mae.reset_state()
            epoch_loss = 0  # Para calcular a perda média da época
            total_steps = 0

            start_idx = 0  # Inicializando o índice inicial antes do loop
            # Laço manual de treinamento
            for step, batch_size in enumerate(batch_sizes):
                # Definindo os limites de cada batch
                end_idx = start_idx + batch_size

                # Garante que o índice final não ultrapasse o tamanho do dataset
                if end_idx > len(train_data):
                    end_idx = len(train_data)

                # Pegando os dados e labels do batch atual
                batch_data = [example[0] for example in train_data[start_idx:end_idx]]
                batch_targets = [example[1] for example in train_data[start_idx:end_idx]]

                # Aqui combinamos os dicionários para formar o batch correto
                batch_data = {key: tf.stack([example[key] for example in batch_data]) for key in batch_data[0]}

                # Converte targets para tensores
                batch_targets = tf.convert_to_tensor(batch_targets)

                # Executando o passo de treinamento
                with tf.GradientTape() as tape:
                    predictions = model(batch_data, training=True)
                    loss_value = loss_fn(batch_targets, predictions)

                # Backpropagation
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Atualiza a métrica
                metric_mae.update_state(batch_targets, predictions)

                # Atualiza a perda acumulada da época
                epoch_loss += loss_value.numpy()
                total_steps += 1

                # Salvar as métricas de treinamento no CSV
                writer.writerow({'epoch': epoch + 1, 'step': step + 1, 'loss': loss_value.numpy(), 'mae': metric_mae.result().numpy()})

                print(f"Step {step + 1}: batch_size = {batch_size}, loss = {loss_value.numpy():.4f}, mae = {metric_mae.result().numpy():.4f}")

                # Atualiza o índice inicial para o próximo batch
                start_idx = end_idx  # Mover o índice inicial para o final do batch atual

                # Checa se já atingimos o final do dataset
                if start_idx >= len(train_data):
                    break

            # Calcular e armazenar a perda média da época e o MAE
            average_loss = epoch_loss / total_steps
            average_mae = metric_mae.result().numpy()

            loss_history.append(average_loss)
            mae_history.append(average_mae)

            # Validação ao final de cada época
            val_loss, val_mae = model.evaluate(ds_teste.batch(32), verbose=0)
            val_loss_history.append(val_loss)
            val_mae_history.append(val_mae)

            # Salvar as métricas de validação no CSV
            writer.writerow({'epoch': epoch + 1, 'step': 'validation', 'loss': '', 'mae': '', 'val_loss': val_loss, 'val_mae': val_mae})
            print(f"Validation loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

# Chamando o loop customizado de treinamento se atentar para a epochs
custom_training_loop_per_step(model, ds_treino, ds_teste, batch_sizes, epochs=40, model_process=model_process)


model.save('LSTM_model_batch_var.h5')
model.save('LSTM_model_batch_var.keras')


log_file = 'training_log.csv'
data = pd.read_csv(log_file)

#------------ Calculando a média de LOSS e MAE por época --------------
block_size = 1879
los_mae = data[['loss', 'mae']]
los_mae = los_mae.dropna()
los_mae.reset_index(inplace=True)
los_mae['epoch'] = los_mae.index // block_size
# Calcula a média de 'loss' e 'mae' para cada grupo
los_mae = los_mae.groupby('epoch').agg({'loss': 'mean', 'mae': 'mean'}).reset_index(drop=True)
print(los_mae)

#------------- Juntando as médias da validação -------------
val_loss_mae = data[['val_loss', 'val_mae']].dropna().reset_index(drop=True)


# Plotando a função de perda
plt.figure(figsize=(12, 5))

# Gráfico de Perda
plt.subplot(1, 2, 1)
plt.plot(los_mae.index, los_mae['loss'], label='Training Loss')
plt.plot(val_loss_mae.index, val_loss_mae['val_loss'], label='Validation Loss', linestyle='dashed')
plt.title('Loss vs Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Gráfico de MAE
plt.subplot(1, 2, 2)
plt.plot(los_mae.index, los_mae['mae'], label='Training mae')
plt.plot(val_loss_mae.index, val_loss_mae['val_mae'], label='Validation mae', linestyle='dashed')
plt.title('MAE vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (MAE)')
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig('LSTM_resultados_batch_var.png')

plt.show()

# Fazendo predição do modelo para dados de teste
predicoes = model.predict(ds_teste.batch(32))

# Convertendo o dataset de teste em um DataFrame para associar as previsões
dataset_teste_df = lstm_df[lstm_df.index.year >= 2023].copy()

#------------ Funções para avaliação do resultado -----------
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


# Precificando com Black Scholes
dataset_teste_df['call_price_bs'] = black_scholes_call(dataset_teste_df['S'], dataset_teste_df['K'],
                                            dataset_teste_df['r'], dataset_teste_df['sigma'],
                                            dataset_teste_df['T'])


# Com as previsões alinhadas as amostras do dataset de teste,
# adicionamos as previsões como uma nova coluna no DataFrame
dataset_teste_df['call_price_lstm'] = predicoes

# Exibir as primeiras linhas para verificar
dataset_teste_df.head()

dataset_teste_df.to_csv('dataset_teste_df_batch_var.csv', index = True)

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

#---------- MSE -------------
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


#---------- Plot de algumas opções -------------
# ITM -> VALEB731-2024-69.32
# ATM -> VALED908-2023-87.3
# OTM -> VALEI906-2023-90.66

codigo_opcao = ['VALEB731-2024-69.32', 'VALED908-2023-87.3', 'VALEI906-2023-90.66']

for cod_op in codigo_opcao:
    df_opcao = dataset_teste_df[dataset_teste_df['cod_negociacao'] == cod_op]

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

    plt.savefig(f'{cod_op}_batch_var.png')
    # Exibindo o gráfico
    plt.show()

