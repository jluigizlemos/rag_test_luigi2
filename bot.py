import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_PATH = "rag_index.faiss"
DOCS_DIR = "docs"

# load or create FAISS index
def get_retriever():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(EMBED_PATH):
        return FAISS.load_local(".", embeddings).as_retriever()
    else:
        vectorstore = FAISS.from_documents([], embeddings)
        vectorstore.save_local(".", EMBED_PATH)
        return vectorstore.as_retriever()

def add_docs_to_index(docs):
    embeddings = OpenAIEmbeddings()
    if os.path.exists(EMBED_PATH):
        vectorstore = FAISS.load_local(".", embeddings)
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(".", EMBED_PATH)

retriever = get_retriever()
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Olá! Envie um arquivo .txt ou .pdf para indexar, ou faça uma pergunta.")

async def handle_docs(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    file = update.message.document
    ext = os.path.splitext(file.file_name)[1].lower()
    if ext not in [".txt", ".pdf"]:
        await update.message.reply_text("Arquivo não suportado. Envie apenas .txt ou .pdf.")
        return
    os.makedirs(DOCS_DIR, exist_ok=True)
    path = f"{DOCS_DIR}/{file.file_name}"
    try:
        await file.get_file().download_to_drive(path)
        if ext == ".txt":
            loader = TextLoader(path)
        else:
            loader = PyPDFLoader(path)
        docs = loader.load()
        add_docs_to_index(docs)
        global retriever, qa
        retriever = get_retriever()
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
        await update.message.reply_text(f"Arquivo indexado com sucesso: {file.file_name} (total de {len(docs)} documentos)")
    except Exception as e:
        await update.message.reply_text(f"Erro ao processar o arquivo: {str(e)}")

async def answer(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    try:
        response = qa.run(question)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Erro ao buscar resposta: {str(e)}")

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.Document.ALL, handle_docs))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer))
app.run_polling()
