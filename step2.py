# etape2.py — Charger et découper un document

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Charger le document
loader = PyPDFLoader("document.pdf")   # mets le nom de ton fichier
documents = loader.load()
print(f"✅ Document chargé : {len(documents)} page(s)")

# 2. Découper en morceaux (chunks)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # ~1000 caractères par morceau
    chunk_overlap=150     # chevauchement pour ne pas couper une idée en deux
)
chunks = splitter.split_documents(documents)
print(f"✅ Document découpé en {len(chunks)} morceaux")

# 3. Regarder le premier morceau pour comprendre
print("\n--- Aperçu du 1er morceau ---")
print(chunks[0].page_content[:300])
