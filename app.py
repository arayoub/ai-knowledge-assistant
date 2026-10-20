# app.py — Interface web du RAG avec Streamlit
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🧠")
st.title("🧠 AI Knowledge Assistant")
st.caption("Pose une question sur ton document — réponses basées sur son contenu (RAG)")

# Charger les composants une seule fois (mise en cache)
@st.cache_resource
def charger():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
        api_version="2024-10-21",
    )
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version="2024-10-21",
    )
    return vectorstore, llm

vectorstore, llm = charger()

# Zone de question
question = st.text_input("Ta question :", placeholder="De quoi parle ce document ?")

if st.button("Demander") and question:
    with st.spinner("Recherche dans le document..."):
        docs = vectorstore.similarity_search(question, k=3)
        contexte = "\n\n".join([d.page_content for d in docs])
        prompt = f"""Réponds à la question en te basant UNIQUEMENT sur le contexte ci-dessous.
Si la réponse n'est pas dans le contexte, dis-le.

CONTEXTE :
{contexte}

QUESTION : {question}
"""
        reponse = llm.invoke(prompt)

    st.subheader("💬 Réponse")
    st.write(reponse.content)

    with st.expander("📄 Voir les extraits utilisés"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Extrait {i} :**")
            st.text(d.page_content[:400] + "...")