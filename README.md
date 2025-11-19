## Proyecto RAG — LangChain + Pinecone
Implementación de un sistema RAG (Retrieval-Augmented Generation) que usa embeddings de Hugging Face, Pinecone como vector DB y un modelo HuggingFace para generación. Incluye: indexación, chunking, upsert en Pinecone, recuperación, cadena RAG, un agente simple tipo tool-based y evaluación por scores.

## 📂 Estructura del Proyecto

```text
.
├─ .env.example
├─ README.md
├─ requirements.txt
├─ rag_app.py
├─ rag_config.py
├─ rag_pipeline.py
├─ retriever_rag.py
├─ embeddings.py
├─ create_vector_store.py
├─ p.txt
├─ evaluation.py
└─ .gitignore
```

## 🚀 Arquitectura y Componentes del Proyecto

A continuación se describe la arquitectura lógica del sistema RAG desarrollado.
El flujo completo sigue una secuencia clara: **carga → procesamiento → indexación → recuperación → generación → evaluación**.

---

## 🧩 Arquitectura del Sistema (Descripción Lógica)

### 1. **Carga de documentos**
Se leen archivos desde `p.txt` o desde `data/docs/*.txt`.

### 2. **Chunking (fragmentación del texto)**
Los documentos se dividen en segmentos pequeños utilizando  
`RecursiveCharacterTextSplitter`, permitiendo una mejor indexación semántica.

### 3. **Embeddings**
Cada chunk se convierte en un vector de 1024 dimensiones mediante el modelo:  
**`intfloat/multilingual-e5-large`** (Hugging Face).

### 4. **Base de Datos Vectorial (Vector Store)**
Los vectores se almacenan en **Pinecone Serverless**, utilizando:
- `PineconeVectorStore`
- Se realiza *upsert* para insertar o actualizar embeddings.

### 5. **Recuperación (Retrieval)**
Cuando llega una pregunta:

1. La consulta se convierte en embedding.  
2. Pinecone ejecuta una búsqueda semántica utilizando *cosine similarity*.  
3. Devuelve los **K chunks más relevantes**.

### 6. **Generación (RAG)**
El modelo generativo **`google/flan-t5-base`** utiliza el contexto recuperado para crear una respuesta coherente y precisa.

### 7. **Agente (tool-based, opcional)**
Incluye un agente simple que puede:
- ✔ Reescribir la pregunta  
- ✔ Ejecutar búsquedas  
- ✔ Generar la respuesta final usando herramientas internas  

### 8. **Evaluación**
El módulo de evaluación calcula métricas como:
- Puntaje de similitud  
- Exactitud del retrieval  

Los resultados pueden exportarse a **`evaluation.csv`**.

---

## 📁 Componentes principales (archivos)

| Archivo | Descripción |
|--------|-------------|
| `rag_app.py` | Script principal: carga docs, hace chunking, upserts, inicializa LLM y el REPL para consultas. |
| `embeddings.py` | (Opcional) Función para inicializar el modelo de embeddings. |
| `create_vector_store.py` | (Opcional) Gestión de creación del vector store. |
| `retriever_rag.py` | Utilidades de retrieval y adaptación a retriever. |
| `evaluation.py` | Corre consultas masivas y genera `evaluation.csv`. |
| `p.txt` | Base de conocimiento usada para indexar. |
| `requirements.txt` | Dependencias del proyecto. |

---

## 🔧 Requisitos y entorno

- Python **3.10+** (recomendado **3.11**)  
- Espacio libre (los modelos de HuggingFace pueden ocupar varios GB en caché)  
- Conexión a Internet (descarga de modelos y acceso a Pinecone)  
- (Opcional) GPU para acelerar la inferencia  

---

## 🔐 Variables de entorno necesarias

```env
PINECONE_API_KEY=pcsk_xxx     
PINECONE_HOST=https://proyecto-xxxx.svc.aped-4627-b74a.pinecone.io
PINECONE_INDEX=proyecto
HUGGINGFACEHUB_API_TOKEN=hf_xxx

