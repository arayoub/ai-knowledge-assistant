# step3.py — Créer les embeddings et les stocker dans FAISS
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()  # charge les secrets du .env

# 1. Charger + découper (comme étape 2)
loader = PyPDFLoader("document.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print(f"✅ {len(chunks)} morceaux prêts")

# 2. Configurer les embeddings Azure
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
    api_version="2024-10-21",
)

# 3. Créer les vecteurs + stocker dans FAISS
print("⏳ Création des embeddings... (quelques secondes)")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Sauvegarder la base sur ton disque
vectorstore.save_local("faiss_index")
print("✅ Base vectorielle créée et sauvegardée dans 'faiss_index/'")