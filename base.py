import os
import requests
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# LangChain imports (latest 2025 structure)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI  # Free Gemini API

# ----------------------------- CONFIGURATION -----------------------------
# Use environment variables for security (required on deployment)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model choice: gemini-1.5-flash is fast & has higher free-tier limits
# Switch to "gemini-1.5-pro" if you enable billing for smarter responses
LLM_MODEL = "gemini-1.5-flash"

# Local embedding model (runs on CPU, no cost)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Flask app for webhook
flask_app = Flask(__name__)

# Telegram bot application
bot_app = Application.builder().token(TELEGRAM_TOKEN).build()

# -------------------------------------------------------------------------

def fetch_repo_contents(owner: str, repo: str, path: str = "") -> str:
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        return None
    
    contents = []
    for item in response.json():
        if item["type"] == "file":
            # Skip large files (>500KB) and obvious binaries
            if item["size"] > 500_000 or item["name"].split(".")[-1].lower() in {
                "png", "jpg", "jpeg", "gif", "bin", "exe", "zip", "pdf", "svg", "ico"
            }:
                continue
            file_resp = requests.get(item["download_url"])
            if file_resp.status_code == 200:
                contents.append(f"\n--- File: {item['path']} ---\n{file_resp.text}")
        elif item["type"] == "dir":
            sub_content = fetch_repo_contents(owner, repo, item["path"])
            if sub_content:
                contents.append(sub_content)
    
    return "\n".join(contents)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I'm your GitHub Repo Chat bot powered by Google Gemini 🤖\n\n"
        "Use /add_repo <github_url> to load a public repository.\n"
        "Example: /add_repo https://github.com/karpathy/micrograd\n\n"
        "Then ask any questions about the code!"
    )

async def add_repo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Please provide a GitHub repo URL.\nExample: /add_repo https://github.com/user/repo")
        return
    
    repo_url = context.args[0]
    if "github.com" not in repo_url:
        await update.message.reply_text("Invalid GitHub URL.")
        return
    
    try:
        parts = repo_url.rstrip("/").split("/")
        owner, repo = parts[-2], parts[-1]
        if "." in repo:  # Handle .git suffix
            repo = repo.split(".")[0]
    except Exception:
        await update.message.reply_text("Could not parse the repo URL.")
        return
    
    await update.message.reply_text(f"Fetching {owner}/{repo}... This may take 30–60 seconds for larger repos.")
    
    full_content = fetch_repo_contents(owner, repo)
    if not full_content:
        await update.message.reply_text("Failed to fetch repository. It might be private, empty, or too large.")
        return
    
    # Intelligent chunking with overlap
    chunk_size = 1000
    chunk_overlap = 200
    texts = []
    start = 0
    while start < len(full_content):
        end = start + chunk_size
        texts.append(full_content[start:end])
        start = end - chunk_overlap
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        google_api_key=GEMINI_API_KEY,
        model=LLM_MODEL,
        temperature=0.3,
    )
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert software engineer analyzing code from a GitHub repository.
        Answer the question clearly and concisely using only the provided context.
        If you cannot answer based on the context, say "Not enough information in the loaded repo."

        Context:
        {context}

        Question: {question}

        Answer:"""
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Store per-user
    context.user_data["rag_chain"] = rag_chain
    context.user_data["repo"] = f"{owner}/{repo}"
    
    await update.message.reply_text(f"✅ Repository {owner}/{repo} loaded successfully!\nYou can now ask questions about the code.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "rag_chain" not in context.user_data:
        await update.message.reply_text("No repository loaded yet. Use /add_repo <url> first.")
        return
    
    question = update.message.text
    await update.message.reply_text("Gemini is thinking...")
    
    try:
        response = context.user_data["rag_chain"].invoke(question)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}\n(This may be due to Gemini rate limits on the free tier.)")

# Add handlers
bot_app.add_handler(CommandHandler("start", start))
bot_app.add_handler(CommandHandler("add_repo", add_repo))
bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Webhook endpoint for deployment
@flask_app.route('/', methods=['GET', 'POST'])
def webhook():
    if request.method == 'POST':
        update = Update.de_json(request.get_json(force=True), bot_app.bot)
        if update:
            bot_app.process_update(update)
        return 'OK', 200
    return 'GitHub Repo Chat Bot is running!', 200

# For local testing only
if __name__ == '__main__':
    print("Running locally with polling (for testing only)...")
    bot_app.run_polling()
    # For deployment on Render/ Railway/ etc., comment out run_polling() and use Flask
    # flask_app.run(host='0.0.0.0', port=int(os.getenv("PORT", 8080)))