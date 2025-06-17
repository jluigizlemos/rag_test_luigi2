import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from langchain.document_loaders import TextLoader
import pdfplumber
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = "chroma_index"
DOCS_DIR = "docs"

# Função para carregar PDF usando pdfplumber
def load_pdf_with_pdfplumber(path):
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(page_content=text, metadata={"source": path, "page": i+1}))
    return docs

# Global variables for QA system
vectorstore = None
qa_chain = None # Renamed from qa to qa_chain for clarity

# Function to initialize or reload the QA system
def initialize_qa_system():
    global vectorstore, qa_chain
    embeddings = OpenAIEmbeddings()
    # Load existing vectorstore or create if it doesn't exist
    # Chroma automatically creates if directory is empty or doesn't exist,
    # and loads if it exists.
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    # Ensure the retriever is created from the potentially updated vectorstore
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
    print("QA system initialized/reloaded.")

# Function to add documents to the index and update QA chain
def add_docs_to_index(docs):
    global vectorstore, qa_chain # Ensure qa_chain is updated
    if vectorstore is None:
        print("Vectorstore not initialized. Call initialize_qa_system() first.")
        # Or initialize it here if preferred, though startup is better
        initialize_qa_system()

    vectorstore.add_documents(docs)
    # Re-initialize the chain to ensure it uses the latest state of the retriever/vectorstore
    # This is important because the number of documents in the store has changed.
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
    print(f"Docs added to index. QA chain updated. Total documents in store: {vectorstore._collection.count()}")

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
        telegram_file = await file.get_file()
        await telegram_file.download_to_drive(path)
        if ext == ".txt":
            loader = TextLoader(path)
            docs = loader.load()
        else:
            docs = load_pdf_with_pdfplumber(path)
        add_docs_to_index(docs) # This will now use the refactored version
        # global retriever, qa # Removed
        # retriever = get_retriever() # Removed
        # qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever) # Removed
        await update.message.reply_text(f"Arquivo indexado com sucesso: {file.file_name} (total de {len(docs)} documentos)")
    except Exception as e:
        error_message = f"Sorry, I couldn't process your file {file.file_name}. Please try again or check the file format."
        print(f"Error processing file {file.file_name}: {e}") # For server-side logging
        await update.message.reply_text(error_message)

async def answer(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global qa_chain
    if qa_chain is None:
        # This is a fallback, should ideally be initialized at startup
        initialize_qa_system()
        if qa_chain is None: # If still None after trying to initialize
            await update.message.reply_text("The question answering system is not ready. Please try again shortly.")
            return

    question = update.message.text
    try:
        response = qa_chain.run(question) # Use qa_chain
        await update.message.reply_text(response)
    except Exception as e:
        error_message = "Sorry, I encountered an issue trying to answer your question. Please try rephrasing or ask something else."
        print(f"Error generating answer for question '{question}': {e}")
        await update.message.reply_text(error_message)

async def list_files(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        await update.message.reply_text("No files have been indexed yet. Send a .txt or .pdf to get started!")
        return

    try:
        indexed_files = [f for f in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, f))]
        if not indexed_files:
            await update.message.reply_text("No files have been indexed yet. The 'docs' directory is empty or contains only subdirectories.")
            return

        message = "Here are the indexed files:\n"
        for file_name in indexed_files:
            message += f"- {file_name}\n"
        await update.message.reply_text(message)
    except Exception as e:
        print(f"Error listing files: {e}")
        await update.message.reply_text("Sorry, I encountered an error while trying to list the indexed files.")

# Initialize the QA system at startup
initialize_qa_system()

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("list_files", list_files))
app.add_handler(MessageHandler(filters.Document.ALL, handle_docs))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer))
app.run_polling()
