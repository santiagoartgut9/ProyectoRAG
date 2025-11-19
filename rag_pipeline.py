from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from retriever_rag import get_retriever

# Cargar LLM
MODEL = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)

retriever = get_retriever()

def rag_query(pregunta):

    documentos = retriever.get_relevant_documents(pregunta)

    contexto = "\n\n".join([doc.page_content for doc in documentos])

    prompt = f"""
Eres un asistente RAG.
Utiliza UNICAMENTE esta información para responder:

Contexto:
{contexto}

Pregunta:
{pregunta}

Respuesta:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.2
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
