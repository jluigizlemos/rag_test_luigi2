# RAG Telegram Bot

Este projeto é um bot de Telegram com RAG (Retrieval-Augmented Generation) usando LangChain, FAISS e OpenAI.

## Rodando Localmente (sem Docker)

### 1. Clone o repositório
```bash
git clone <url-do-repo>
cd RAG _test
```

### 2. Crie um ambiente virtual Python
```bash
python -m venv .venv
```

### 3. Ative o ambiente virtual
- **Linux/macOS:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```

### 4. Instale as dependências
```bash
pip install -r requirements.txt
```

### 5. Configure as variáveis de ambiente
Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:
```
TELEGRAM_TOKEN=seu_token_do_telegram
OPENAI_API_KEY=sua_chave_openai
```

### 6. Execute o bot
```bash
python bot.py
```

---

## Deploy em Nuvem (exemplo: Railway)

1. Suba seu projeto para um repositório no GitHub.
2. Crie uma conta em [Railway](https://railway.app/).
3. Clique em "New Project" > "Deploy from GitHub repo" e selecione seu repositório.
4. No painel do projeto, vá em "Variables" e adicione:
   - `TELEGRAM_TOKEN` com o token do seu bot Telegram
   - `OPENAI_API_KEY` com sua chave da OpenAI
5. Railway detecta o `requirements.txt` automaticamente. Se necessário, defina o comando de start como:
   ```
   python bot.py
   ```
6. Clique em "Deploy". O bot ficará rodando na nuvem!

---

## Observações
- Os arquivos enviados ao bot serão salvos na pasta `docs/`.
- O índice Chroma é salvo na pasta `chroma_index/` na raiz do projeto.
- Suporte a arquivos `.txt` e `.pdf`.

---

## Requisitos
- Python 3.10+
- Conta no Telegram para criar um bot e obter o token
- Chave de API da OpenAI
