# GitChat
An AI-powered Telegram bot that lets you chat with any public GitHub repository using Retrieval-Augmented Generation (RAG).Load a repo and ask natural language questions about its code — get accurate, context-aware answers powered by Google Gemini!

 FeaturesLoad any public GitHub repo with a simple command
Ask questions about code, architecture, functions, and more
True RAG architecture for grounded, hallucination-free responses
Local embeddings with Hugging Face + FAISS for fast retrieval
Powered by Google Gemini 1.5 Flash (free tier)
Per-user sessions for private conversations
Easy deployment (Render.com, Railway, etc.)

 How It Works (RAG Architecture)The bot uses Retrieval-Augmented Generation to provide precise answers based on the actual repo code:gradientflow.substack.com

Fetch & chunk repo files via GitHub API  
Generate embeddings locally (sentence-transformers)  
Store in FAISS vector database  
On query: Retrieve relevant chunks → Augment prompt → Generate answer with Gemini

 Demo in Actiongithub.com

Example: Load karpathy/micrograd and ask "Explain backpropagation here" Tech StackPython with python-telegram-bot
LangChain (modern Runnable chains)
Hugging Face for embeddings (all-MiniLM-L6-v2)
FAISS for vector search
Google Gemini API via LangChain
Flask for webhook deployment

** Quick Start (Local)Clone the repo:bash**

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

Install dependencies:bash

pip install -r requirements.txt



export TELEGRAM_TOKEN="your_bot_token"
export GEMINI_API_KEY="your_gemini_key"

Run:bash

python bot.py

Open Telegram and chat with your bot!

 DeploymentDeploy for free on Render.com:Use gunicorn bot:flask_app as start command




