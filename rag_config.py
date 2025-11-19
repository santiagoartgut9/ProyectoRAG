import os
from pinecone import Pinecone

# Leer variables de entorno CORRECTAMENTE
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

print("Pinecone API Key cargada:", bool(PINECONE_API_KEY))
print("HuggingFace Token cargado:", bool(HUGGINGFACE_TOKEN))

# Inicializar cliente Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Nombre del índice
INDEX_NAME = "proyecto"

# Conectar al índice
index = pc.Index(INDEX_NAME)

print("Conexión al índice completada:", INDEX_NAME)
print("Listando índices disponibles:")
print(pc.list_indexes())
