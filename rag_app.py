

import os
import sys
from pathlib import Path
from uuid import uuid4
import time

# Pinecone
from pinecone import Pinecone

# LangChain helpers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Transformers for generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------
# Config (env)
# -----------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST = os.environ.get("PINECONE_HOST")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "proyecto")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Models
EMBED_MODEL = "intfloat/multilingual-e5-large"   # mantiene 1024 dims (asegurar compatibilidad)
LLM_MODEL = "google/flan-t5-base"                 # mejor que small (calidad)
TOP_K = 3

# -----------------------
# Validaciones
# -----------------------
if not PINECONE_API_KEY:
    print("ERROR: falta PINECONE_API_KEY")
    sys.exit(1)
if not PINECONE_HOST:
    print("ERROR: falta PINECONE_HOST")
    sys.exit(1)
if not HUGGINGFACE_TOKEN:
    print("ERROR: falta HUGGINGFACEHUB_API_TOKEN")
    sys.exit(1)

# -----------------------
# Conexión a Pinecone (serverless usando host)
# -----------------------
print("Conectando a Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
try:
    index = pc.Index(host=PINECONE_HOST)
    print("Conectado al índice host:", PINECONE_HOST)
except Exception as e:
    print("ERROR conectando a Pinecone:", e)
    sys.exit(1)

# -----------------------
# Utils: cargar documentos (UTF-8 safe)
# -----------------------
def load_documents():
    docs = []
    data_dir = Path("data/docs")
    if data_dir.exists() and any(data_dir.glob("*.txt")):
        for p in sorted(data_dir.glob("*.txt")):
            txt = p.read_text(encoding="utf-8", errors="replace")
            docs.append(Document(page_content=txt, metadata={"source": str(p)}))
        print(f"Cargados {len(docs)} documentos desde data/docs/")
    else:
        fallback = Path("p.txt")
        if fallback.exists():
            txt = fallback.read_text(encoding="utf-8", errors="replace")
            docs.append(Document(page_content=txt, metadata={"source": "p.txt"}))
            print("Cargado fallback p.txt")
        else:
            docs = [
                Document(page_content="LangChain es un framework para construir aplicaciones con LLMs.", metadata={"source":"sample_1"}),
                Document(page_content="Pinecone es una base de datos vectorial serverless para vectores densos.", metadata={"source":"sample_2"}),
            ]
            print("No se encontraron archivos: usando 2 ejemplos.")
    return docs

# -----------------------
# Chunking
# -----------------------
def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    splits = splitter.split_documents(docs)
    print(f"Documentos divididos en {len(splits)} chunks.")
    return splits

# -----------------------
# Crear embeddings & upsert (devuelve vector_store y embeddings_inst)
# -----------------------
def create_and_upsert(splits):
    print("Inicializando embeddings:", EMBED_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device":"cpu"}, encode_kwargs={"normalize_embeddings": True})
    # Construir vector store ligado al index (serverless host)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    ids = [str(uuid4()) for _ in splits]
    print(f"Upserting {len(splits)} vectores en Pinecone (index host)...")
    vector_store.add_documents(documents=splits, ids=ids)
    print("Upsert completado.")
    return vector_store, embeddings

# -----------------------
# Cargar LLM generador
# -----------------------
def load_generation_model():
    print("Cargando modelo de generación:", LLM_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Modelo cargado en:", device)
    return tokenizer, model, device

# -----------------------
# Generación: prompt robusto
# -----------------------
def generate_with_context(question, docs_text, tokenizer, model, device, max_new_tokens=256):
    prompt = (
        "Eres un asistente experto. Usa SOLO la información del contexto a continuación para responder de forma clara, concisa y sin repetir textualmente.\n\n"
        f"Contexto:\n{docs_text}\n\n"
        f"Pregunta: {question}\n\n"
        "Respuesta (en español, si la pregunta está en español). "
        "Cita la(s) fuente(s) si es posible al final, y sé breve.\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# -----------------------
# RAG: recuperar + generar
# -----------------------
def rag_chain_answer(question, vector_store, tokenizer, model, device, top_k=TOP_K):
    retrieved = vector_store.similarity_search(question, k=top_k)
    if not retrieved:
        return "No se encontraron documentos relevantes."
    docs_text = "\n\n".join([f"[{d.metadata}] {d.page_content}" for d in retrieved])
    return generate_with_context(question, docs_text, tokenizer, model, device)

# -----------------------
# Agente simple (tool-based) -> genera query, recupera y responde
# -----------------------
def generate_search_query(question, tokenizer, model, device):
    """
    Usamos el LLM para reescribir/condensar la pregunta en una buena 'search query'.
    """
    prompt = f"Escribe una consulta corta (5-12 palabras) para buscar información relevante para la siguiente pregunta:\n\n{question}\n\nConsulta:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    q = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return q

def agent_answer(question, vector_store, tokenizer, model, device, top_k=TOP_K):
    # 1) Generar query de búsqueda
    search_q = generate_search_query(question, tokenizer, model, device)
    if not search_q:
        search_q = question
    # 2) Recuperar con la query
    retrieved = vector_store.similarity_search(search_q, k=top_k)
    # 3) Si es necesario, concatenar también recuperación con la pregunta original
    combined = "\n\n".join([f"[{d.metadata}] {d.page_content}" for d in retrieved])
    # 4) Generar respuesta usando contexto recuperado
    return generate_with_context(question, combined, tokenizer, model, device)

# -----------------------
# Evaluación / métricas (usa embeddings.inst para calcular query vector)
# -----------------------
def evaluate_retrieval(question, embeddings_inst, top_k=5):
    print("\n=== Evaluación retrieval ===")
    q_vec = embeddings_inst.embed_query(question)
    res = index.query(vector=q_vec, top_k=top_k, include_metadata=True, include_values=False)
    matches = res.get("matches", [])
    if not matches:
        print("No matches")
        return
    for i, m in enumerate(matches, 1):
        score = m.get("score", 0.0)
        meta = m.get("metadata", {})
        text = meta.get("text") or meta.get("source") or str(meta)  # intento de mostrar algo útil
        print(f"{i}. score={score:.4f} - meta={text[:150]}")
    print("===========================\n")

# -----------------------
# MAIN
# -----------------------
def main():
    # 1) Cargar docs
    docs = load_documents()
    splits = chunk_documents(docs)

    # 2) Embeddings + upsert
    vector_store, embeddings_inst = create_and_upsert(splits)

    # 3) Cargar LLM
    tokenizer, model, device = load_generation_model()

    print("\n--- RAG listo ---")
    print("Escribe preguntas (o 'exit')\n")

    while True:
        q = input("Pregunta: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit", "salir"):
            print("Bye.")
            break

        # RAG chain
        print("\n--- RESPUESTA (RAG chain) ---")
        try:
            ans = rag_chain_answer(q, vector_store, tokenizer, model, device)
            print(ans)
        except Exception as e:
            print("Error en RAG chain:", e)

        # Agent (tool-based simple)
        print("\n--- RESPUESTA (Agent tool-based) ---")
        try:
            ans2 = agent_answer(q, vector_store, tokenizer, model, device)
            print(ans2)
        except Exception as e:
            print("Error en agent:", e)

        # Evaluación / scores
        try:
            evaluate_retrieval(q, embeddings_inst, top_k=5)
        except Exception as e:
            print("Error en evaluación:", e)

        print("\n" + ("-" * 60) + "\n")


if __name__ == "__main__":
    main()
