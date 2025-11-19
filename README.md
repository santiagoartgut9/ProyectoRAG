
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Embeddings](https://img.shields.io/badge/Embeddings-multilingual--e5--large-green)
![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-blueviolet)
![FLAN T5](https://img.shields.io/badge/LLM-FLAN--T5--Base-lightgrey)
![Torch](https://img.shields.io/badge/Backend-Torch-orange)



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
```

## 🛠️ Instalación (paso a paso)

### 1) Crear entorno virtual

#### 🪟 Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
🐧 Linux / 🍎 macOS
```
python -m venv .venv
source .venv/bin/activate

```
## Instalar dependencias

Guarda las dependencias en requirements.txt y luego instala:
```
pip install -r requirements.txt
```

📌 requirements.txt
```
langchain>=0.2.10
langchain-pinecone
pinecone-client>=0.6.0
langchain-huggingface
langchain-text-splitters
langchain-core
transformers>=4.30.0
sentence-transformers
torch
huggingface-hub
bs4

```
## Exportar variables de entorno
```
$env:PINECONE_API_KEY = "pcsk_..."
$env:PINECONE_HOST = "https://proyecto-xxxx.svc.aped-4627-b74a.pinecone.io"
$env:PINECONE_INDEX = "proyecto"
$env:HUGGINGFACEHUB_API_TOKEN = "hf_..."
```
## 📝Evidencias:


<img width="858" height="379" alt="image" src="https://github.com/user-attachments/assets/bcbf8fd4-4168-4c39-abef-616fda4ab461" />

<img width="1350" height="600" alt="image" src="https://github.com/user-attachments/assets/b7f2967b-92c6-4f3f-9764-338d764cfd8c" />

<img width="967" height="311" alt="image" src="https://github.com/user-attachments/assets/ac08ef1a-a48e-48b6-8591-4f3e38ff6321" />

<img width="827" height="257" alt="image" src="https://github.com/user-attachments/assets/5f2411cd-0ac9-497d-a17f-823cd7a10247" />

<img width="821" height="299" alt="image" src="https://github.com/user-attachments/assets/cb260f14-745c-4fff-974a-e8f3b6af0b84" />

<img width="652" height="383" alt="image" src="https://github.com/user-attachments/assets/1051a871-b15a-4ef8-80ec-fdec5c6a5eab" />

<img width="414" height="283" alt="image" src="https://github.com/user-attachments/assets/43fbbc05-197b-4643-b706-2be2cff79047" />

<img width="465" height="93" alt="image" src="https://github.com/user-attachments/assets/a028c64d-ffb8-411a-8ae6-179ebf031d9e" />

<img width="430" height="182" alt="image" src="https://github.com/user-attachments/assets/b4c0be20-3eb9-4b64-9773-cc5c88d98ab9" />

<img width="978" height="171" alt="image" src="https://github.com/user-attachments/assets/c5b72cc3-d6df-417e-8104-1a68118c1eca" />

<img width="494" height="120" alt="image" src="https://github.com/user-attachments/assets/fd68e53b-3d7a-4547-8660-cf7ec50ad344" />

<img width="953" height="209" alt="image" src="https://github.com/user-attachments/assets/cca13ca0-cf20-4b96-95c1-e64565fc41f1" />

<img width="975" height="293" alt="image" src="https://github.com/user-attachments/assets/c668be18-cf2f-4b67-9ca9-cf074e5feabd" />

<img width="933" height="323" alt="image" src="https://github.com/user-attachments/assets/c72b37b4-cf33-40e3-a48e-c3326034565e" />

















