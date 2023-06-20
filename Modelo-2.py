import os
import sqlite3
import nltk
import tkinter as tk
import requests
import json
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Baixe os recursos necessários do NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Importe as stop words em português do NLTK
stop_words = set(stopwords.words('portuguese'))

# Carregar o modelo BERTimbau e o tokenizador
model_name_bert = "neuralmind/bert-base-portuguese-cased"
model_bert = BertForSequenceClassification.from_pretrained(model_name_bert)
tokenizer_bert = BertTokenizer.from_pretrained(model_name_bert)

# Carregar o modelo GPT-2 base e o tokenizador
model_name_gpt2 = "gpt2"
model_gpt2 = GPT2LMHeadModel.from_pretrained(model_name_gpt2)
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(model_name_gpt2)

# Conectar ao banco de dados SQLite
con = sqlite3.connect('chatbot.db')
cur = con.cursor()

# Criar a tabela para armazenar as perguntas e respostas
cur.execute('''
    CREATE TABLE IF NOT EXISTS chat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pergunta TEXT,
        resposta_modelo TEXT,
        resposta_google TEXT
    )
''')

# Função para processar as perguntas e respostas e armazená-las no banco de dados
def processar_perguntas_respostas(pergunta, resposta_modelo, resposta_google):
    # Remover as stop words da pergunta antes de armazená-la no banco de dados
    pergunta_sem_stopwords = ' '.join([word for word in pergunta.split() if word.lower() not in stop_words])

    cur.execute('''
        INSERT INTO chat (pergunta, resposta_modelo, resposta_google)
        VALUES (?, ?, ?)
    ''', (pergunta_sem_stopwords, resposta_modelo, resposta_google))
    con.commit()

# Função para gerar uma resposta com o modelo GPT-2
def gerar_resposta(pergunta):
    perguntas_tokenizadas = tokenizer_gpt2.encode(pergunta, return_tensors="pt")
    resposta = model_gpt2.generate(perguntas_tokenizadas, max_length=100, num_return_sequences=1)
    resposta_decodificada = tokenizer_gpt2.decode(resposta[0], skip_special_tokens=True)

    # Armazenar a pergunta, resposta do modelo GPT-2 e resposta do Google no banco de dados
    processar_perguntas_respostas(pergunta, resposta_decodificada, None)

    return resposta_decodificada

# Função para fazer uma requisição à API de Pesquisa do Google
def pesquisar_no_google(pergunta):
    # Configurar a URL da API de Pesquisa do Google
    url = "https://www.googleapis.com/customsearch/v1"
    api_key = "AIzaSyBL7rLkqEMc1hWST9je2RnG6cV_qZwcjok"
    cx = "b696f23d7a3ea4aff"
    num_results = 1  # Número de resultados desejados

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

            # Armazenar a pergunta, resposta do modelo GPT-2 e resposta do Google no banco de dados
            processar_perguntas_respostas(pergunta, None, resposta)

            return resposta
        else:
            return "Desculpe, não consegui obter uma resposta no momento."
    except requests.exceptions.RequestException as e:
        print("Ocorreu um erro durante a requisição:", str(e))
        return "Desculpe, ocorreu um erro durante a requisição."

# Função para atualizar e re-treinar o modelo GPT-2 com base no histórico de perguntas e respostas
def atualizar_e_retreinar_modelo():
    cur.execute('SELECT pergunta, resposta_modelo FROM chat')
    registros = cur.fetchall()

    perguntas = []
    respostas = []

    for registro in registros:
        pergunta = registro[0]
        resposta = registro[1]

        perguntas.append(pergunta)
        respostas.append(resposta)

    # Re-treinar o modelo GPT-2 com os dados atualizados
    perguntas_tokenizadas = tokenizer_gpt2.batch_encode_plus(
        perguntas,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )["input_ids"]

    respostas_tokenizadas = tokenizer_gpt2.batch_encode_plus(
        respostas,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )["input_ids"]

    model_gpt2.train()

    # Treinar o modelo com os dados atualizados
    model_gpt2.per_device_train_batch_size = 4
    model_gpt2.gradient_accumulation_steps = 16
    model_gpt2.num_train_epochs = 2
    model_gpt2.learning_rate = 1e-5

    # Exemplo de loop de treinamento
    for epoch in range(model_gpt2.num_train_epochs):
        # Treinar o modelo em lotes
        for i in range(0, len(perguntas_tokenizadas), model_gpt2.per_device_train_batch_size):
            input_ids = perguntas_tokenizadas[i:i + model_gpt2.per_device_train_batch_size].to(model_gpt2.device)
            labels = respostas_tokenizadas[i:i + model_gpt2.per_device_train_batch_size].to(model_gpt2.device)

            outputs = model_gpt2(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()

            # Atualizar os pesos do modelo
            optimizer.step()
            model_gpt2.zero_grad()

    # Salvar o modelo atualizado
    model_gpt2.save_pretrained(model_name_gpt2)
    tokenizer_gpt2.save_pretrained(model_name_gpt2)

    print("Modelo GPT-2 atualizado e re-treinado com sucesso.")

# Função para buscar na documentação do Python
def buscar_documentacao(pergunta):
    # Exemplo de busca na documentação do Python
    url = "https://docs.python.org/3/search.html?q=" + pergunta.replace(" ", "+")

    # Fazer a requisição HTTP
    response = requests.get(url)

    if response.status_code == 200:
        # Extrair o conteúdo HTML da resposta
        html = response.text

        # Parsear o HTML usando o BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Extrair os trechos relevantes da documentação
        trechos = soup.find_all("p")

        # Retornar o primeiro trecho encontrado
        if trechos:
            return trechos[0].text
        else:
            return "Não foi possível encontrar uma resposta na documentação."
    else:
        return "Não foi possível acessar a documentação no momento."

# Interface gráfica
janela = tk.Tk()
janela.title("Chatbot")
janela.geometry("400x500")

texto_resposta = tk.Text(janela, width=50, height=20)
texto_resposta.config(state=tk.DISABLED)
texto_resposta.pack(pady=10)

entrada_pergunta = tk.Entry(janela, width=30)
entrada_pergunta.pack(pady=10)

historico_perguntas = []
historico_respostas_modelo = []
historico_respostas_google = []

def processar_pergunta():
    pergunta = entrada_pergunta.get()

    # Atualizar o histórico de perguntas
    historico_perguntas.append(pergunta)

    # Verificar o contexto anterior
    contexto_anterior = ""
    if len(historico_perguntas) > 1:
        contexto_anterior = historico_perguntas[-2]

    resposta_modelo = gerar_resposta(pergunta)
    resposta_google = pesquisar_no_google(pergunta)
    resposta_documentacao = buscar_documentacao(pergunta)

    # Verificar se a resposta do modelo GPT-2 é semelhante ao contexto anterior
    if resposta_modelo.lower() in contexto_anterior.lower():
        resposta_modelo = "A resposta do modelo é semelhante ao contexto anterior."

    # Atualizar o histórico de respostas
    historico_respostas_modelo.append(resposta_modelo)
    historico_respostas_google.append(resposta_google)

    # Atualizar o texto da resposta na interface gráfica para mostrar ambas as respostas
    texto_resposta.config(state=tk.NORMAL)
    texto_resposta.delete("1.0", tk.END)
    texto_resposta.insert(tk.END, "Resposta do modelo GPT-2:\n")
    texto_resposta.insert(tk.END, resposta_modelo)
    texto_resposta.insert(tk.END, "\n\nResposta do Google:\n")
    texto_resposta.insert(tk.END, resposta_google)
    texto_resposta.insert(tk.END, "\n\nResposta da documentação:\n")
    texto_resposta.insert(tk.END, resposta_documentacao)
    texto_resposta.config(state=tk.DISABLED)

    # Limpar a entrada
    entrada_pergunta.delete(0, tk.END)

    # Atualizar e re-treinar o modelo GPT-2 com base no histórico de perguntas e respostas
    atualizar_e_retreinar_modelo()

def pesquisar_e_baixar_csv(url):
    nome_arquivo = "dados.csv"

    # Realizar o download do arquivo CSV
    response = requests.get(url)

    # Verificar se o download foi bem-sucedido
    if response.status_code == 200:
        # Salvar o arquivo CSV
        with open(nome_arquivo, "wb") as file:
            file.write(response.content)
        print("Arquivo CSV baixado com sucesso.")
    else:
        print("Não foi possível baixar o arquivo CSV.")

# Exemplo de pergunta e obtenção da resposta
pergunta_exemplo = "Qual é a capital da França?"
resposta_exemplo = gerar_resposta(pergunta_exemplo)
print("Pergunta:", pergunta_exemplo)
print("Resposta do modelo GPT-2:", resposta_exemplo)

# Exemplo de pergunta e obtenção da resposta do Google
pergunta_exemplo_google = "Qual é a fórmula do dióxido de carbono?"
resposta_exemplo_google = pesquisar_no_google(pergunta_exemplo_google)
print("Pergunta:", pergunta_exemplo_google)
print("Resposta do Google:", resposta_exemplo_google)

# Exemplo de pergunta e busca na documentação
pergunta_exemplo_documentacao = "Como fazer um loop em Python?"
resposta_exemplo_documentacao = buscar_documentacao(pergunta_exemplo_documentacao)
print("Pergunta:", pergunta_exemplo_documentacao)
print("Resposta da documentação:", resposta_exemplo_documentacao)

# Exemplo de atualização e re-treinamento do modelo GPT-2
atualizar_e_retreinar_modelo()

janela.mainloop()
