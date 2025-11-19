from langchain_huggingface import HuggingFaceEmbeddings

# Modelo de 1024 dimensiones compatible con tu índice Pinecone
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
