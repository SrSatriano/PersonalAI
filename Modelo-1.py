import pandas as pd
import os
import zipfile
import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tkinter as tk
import requests
from nltk.tokenize import WhitespaceTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TFGPT2LMHeadModel
import torch
import math
from bs4 import BeautifulSoup
import re

# Baixe os recursos necessários do NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Carregar o modelo GPT-2.5-small e o tokenizador
model_name = "distilgpt2"
model = TFGPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Diretório para salvar os arquivos CSV baixados
diretorio_csv = "C:/Users/Satriano/Documents/Programação/Python/IA/IA de uso pessoal/CSVs"

# Função para pesquisar e baixar um arquivo CSV
def pesquisar_e_baixar_csv(termo_pesquisa):
    # TODO: Implemente a pesquisa na web e seleção de uma fonte confiável para download do arquivo CSV
    # Utilize a biblioteca ou API de sua escolha para realizar a pesquisa na web
    # Processar os resultados da pesquisa e selecionar uma fonte confiável com arquivo CSV
    
    # Exemplo de download de arquivo CSV
    url = "http://exemplo.com/arquivo.csv"  # Substitua pela URL do arquivo CSV selecionado
    
    # Realizar o download do arquivo CSV
    try:
        response = requests.get(url)
        response.raise_for_status()  # Verificar se a resposta foi bem-sucedida
        arquivo_csv = os.path.join(diretorio_csv, "arquivo.csv")  # Caminho para salvar o arquivo CSV localmente
        with open(arquivo_csv, "wb") as file:
            file.write(response.content)
        return arquivo_csv
    except requests.exceptions.RequestException as e:
        print("Ocorreu um erro durante o download do arquivo CSV:", str(e))
        return None

# Carregar o conjunto de dados da Kaggle (supondo que você tenha um arquivo CSV chamado "kaggle_dataset.csv" com as colunas "pergunta" e "resposta")
pasta_kaggle = "C:/Users/Satriano/Documents/Programação/Python/IA/IA de uso pessoal/Kaggle"

# Lista para armazenar os dataframes de cada arquivo XLS
dataframes = []

# Função para percorrer recursivamente os diretórios e ler os arquivos
def ler_arquivos(caminho):
    for root, dirs, files in os.walk(caminho):
        for arquivo in files:
            if arquivo.endswith(".xls"):
                caminho_arquivo = os.path.join(root, arquivo)
                df = pd.read_excel(caminho_arquivo)
                print("Colunas disponíveis no dataframe:", arquivo)
                print(df.columns)  # Imprimir as colunas disponíveis no dataframe
                dataframes.append(df)
            elif arquivo.endswith(".parquet"):
                caminho_arquivo = os.path.join(root, arquivo)
                df = pd.read_parquet(caminho_arquivo)
                print("Colunas disponíveis no dataframe:", arquivo)
                print(df.columns)  # Imprimir as colunas disponíveis no dataframe
                dataframes.append(df)
            elif arquivo.endswith(".whl"):
                # Processar arquivo WHL conforme necessário
                pass
            elif arquivo.endswith(".xlsx"):
                caminho_arquivo = os.path.join(root, arquivo)
                df = pd.read_excel(caminho_arquivo)
                print("Colunas disponíveis no dataframe:", arquivo)
                print(df.columns)  # Imprimir as colunas disponíveis no dataframe
                dataframes.append(df)

# Chamar a função para ler os arquivos dentro da pasta Kaggle
ler_arquivos(pasta_kaggle)

# Pesquisar e baixar um arquivo CSV
termo_pesquisa = "exemplo de arquivo CSV"  # Substitua pelo termo de pesquisa desejado
arquivo_csv = pesquisar_e_baixar_csv(termo_pesquisa)

# Ler o arquivo CSV baixado, se disponível
if arquivo_csv:
    df = pd.read_csv(arquivo_csv)
    dataframes.append(df)

# Concatenar os dataframes em um único dataframe
if dataframes:
    kaggle_dataset = pd.concat(dataframes)
    if "pergunta" in kaggle_dataset.columns:
        perguntas_kaggle = kaggle_dataset["pergunta"].values
        respostas_kaggle = kaggle_dataset["resposta"].values
    else:
        perguntas_kaggle = np.array([])
        respostas_kaggle = np.array([])
else:
    perguntas_kaggle = np.array([])
    respostas_kaggle = np.array([])


# Pré-processamento dos dados
lemmatizer = WordNetLemmatizer()

perguntas = np.concatenate([perguntas_kaggle])
respostas = np.concatenate([respostas_kaggle])

tokenized_perguntas = [word_tokenize(pergunta) for pergunta in perguntas]
lemmatized_perguntas = [[lemmatizer.lemmatize(word) for word in tokens] for tokens in tokenized_perguntas]
tokenized_respostas = [word_tokenize(resposta) for resposta in respostas]
lemmatized_respostas = [[lemmatizer.lemmatize(word) for word in tokens] for tokens in tokenized_respostas]

# Construção do modelo
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([" ".join(tokens) for tokens in lemmatized_perguntas + lemmatized_respostas])

modelo = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

# Compilação do modelo
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
print("Número de perguntas:", len(lemmatized_perguntas))
print("Número de respostas:", len(lemmatized_respostas))

if lemmatized_perguntas and lemmatized_respostas:
    modelo.fit(lemmatized_perguntas, lemmatized_respostas, epochs=100)
else:
    print("Não há dados disponíveis para treinar o modelo.")

# Função para gerar uma resposta
def gerar_resposta(pergunta):
    perguntas_tokenizadas = nltk.word_tokenize(pergunta)
    input_ids = tokenizer.encode(" ".join(perguntas_tokenizadas), return_tensors="tf")
    output = model.generate(input_ids, max_length=100)
    resposta = tokenizer.decode(output[0], skip_special_tokens=True)
    return resposta

# Gerar a resposta usando o modelo pré-treinado
    output = model.generate(pergunta_tokenizada, max_length=100)
    resposta = tokenizer.decode(output[0], skip_special_tokens=True)
    return resposta

# Função para fazer uma requisição à API de Pesquisa do Google
def pesquisar_no_google(pergunta):
    # Configurar a URL da API de Pesquisa do Google
    url = "https://www.googleapis.com/customsearch/v1"
    api_key = "721770251957-fhurbfmvk9qmq22sl307v80q3fe0sogt.apps.googleusercontent.com"
    cx = "b696f23d7a3ea4aff"
    num_results = 2  # Número de resultados desejados

    # Parâmetros da requisição
    parametros = {
        "key": api_key,
        "cx": cx,
        "q": pergunta,
        "num": num_results
    }

    try:
        # Fazer a requisição HTTP para o Google Custom Search API
        response = requests.get(url, params=parametros)

        # Verificar se a requisição foi bem-sucedida
        if response.status_code == 200:
            # Extrair a resposta do corpo da resposta HTTP
            resposta = response.json()["items"][0]["snippet"]
            return resposta
        else:
            return "Desculpe, não consegui obter uma resposta no momento."
    except requests.exceptions.RequestException as e:
        print("Ocorreu um erro durante a requisição:", str(e))
        return "Desculpe, ocorreu um erro durante a requisição."

# Interface gráfica
janela = tk.Tk()
janela.title("Chatbot")
janela.geometry("400x500")

texto_resposta = tk.Text(janela, width=50, height=20)
texto_resposta.config(state=tk.DISABLED)
texto_resposta.pack(pady=10)

entrada_pergunta = tk.Entry(janela, width=30)
entrada_pergunta.pack(pady=10)

def processar_pergunta():
    pergunta = entrada_pergunta.get()
    resposta = gerar_resposta(pergunta)
    texto_resposta.config(state=tk.NORMAL)
    texto_resposta.delete("1.0", tk.END)
    texto_resposta.insert(tk.END, resposta)
    texto_resposta.config(state=tk.DISABLED)

    # Exibir resposta no console
    print(resposta)

botao_enviar = tk.Button(janela, text="Enviar", command=processar_pergunta)
botao_enviar.pack(pady=10)

janela.mainloop()
