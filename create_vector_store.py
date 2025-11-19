from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from uuid import uuid4
from rag_config import index
from embeddings import get_embeddings

def create_vector_store():

    embeddings = get_embeddings()

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    # Ejemplo de documentos
    docs = [
        Document(
            page_content="LangChain es un framework para construir aplicaciones con LLMs.",
            metadata={"fuente": "manual"}
        ),
        Document(
            page_content="Pinecone es una base de datos vectorial serverless.",
            metadata={"fuente": "manual"}
        )
    ]

    ids = [str(uuid4()) for _ in docs]

    vector_store.add_documents(documents=docs, ids=ids)

    print("Documentos cargados correctamente.")

if __name__ == "__main__":
    create_vector_store()
