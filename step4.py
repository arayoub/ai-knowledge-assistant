# step4.py — Poser une question au document (RAG, méthode moderne)
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1. Recharger les embeddings (même config qu'à l'étape 3)
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
    api_version="2024-10-21",
)

# 2. Charger la base vectorielle sauvegardée
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
print("✅ Base vectorielle chargée")

# 3. Configurer le LLM (gpt-5-mini)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    api_version="2024-10-21",
)

# 4. La question
question = "De quoi parle ce document ? Résume en 3 points."
print(f"\n❓ Question : {question}\n")

# 5. RETRIEVAL : chercher les 3 morceaux les plus pertinents
docs = vectorstore.similarity_search(question, k=3)
contexte = "\n\n".join([d.page_content for d in docs])

# 6. AUGMENTED GENERATION : donner le contexte + la question au LLM
prompt = f"""Réponds à la question en te basant UNIQUEMENT sur le contexte ci-dessous.
Si la réponse n'est pas dans le contexte, dis-le.

CONTEXTE :
{contexte}

QUESTION : {question}
"""

reponse = llm.invoke(prompt)

print("💬 Réponse :")
print(reponse.content)