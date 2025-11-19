from langchain_pinecone import PineconeVectorStore
from embeddings import get_embeddings
from rag_config import index

def get_retriever():

    embeddings = get_embeddings()

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever
