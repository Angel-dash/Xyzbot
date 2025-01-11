from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from Rag.chunking import split_text_to_chunks

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

all_chunks = split_text_to_chunks()


def generate_embeddings(splits):
    chunks_embeddings = [
        {"text": chunk.page_content, "embeddings": embeddings.embed_query(chunk.page_content)}
        for chunk in splits
    ]
    return chunks_embeddings


def store_embeddings_in_chroma(chunk_embeddings):
    vector_db = Chroma(
        collection_name='transcript_knowledge_base',
        embedding_function=GoogleGenerativeAIEmbeddings(),

    )
    for chunk in chunk_embeddings:
        vector_db.add_texts(chunk['text'], embeddings=chunk['embedding'])
    return vector_db


transcripts_embeddings = generate_embeddings(all_chunks)
