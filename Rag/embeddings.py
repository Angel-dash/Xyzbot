from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from Rag.chunking import split_text_to_chunks
from tqdm import tqdm
import numpy as np
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer

all_chunks = split_text_to_chunks()


def generate_embeddings(splits, batch_size = 32):
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    texts = [chunk.page_content for chunk in splits]
    chunks_embeddings = []
    with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch)
            chunks_embeddings.extend(batch_embeddings)
            pbar.update(len(batch))

    return np.array(chunks_embeddings)


def store_in_chroma(chunks, embeddings):
    # Initialize Chroma client
    client = chromadb.Client(Settings(
        persist_directory="db"  # This will store the database on disk
    ))

    # Create or get collection
    collection = client.create_collection(
        name="transcript_collection",
        metadata={"description": "Video transcript embeddings"}
    )

    # Prepare data for insertion
    ids = [str(i) for i in range(len(chunks))]
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Add data to collection
    with tqdm(total=len(documents), desc="Storing in Chroma") as pbar:
        # You might want to batch this too if dealing with very large datasets
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        pbar.update(len(documents))

    return collection


def main():
    # Get your chunks from your existing code
    all_chunks = split_text_to_chunks()

    print(f"Starting embedding generation for {len(all_chunks)} chunks...")

    # Generate embeddings
    embeddings = generate_embeddings(all_chunks)

    print("Embeddings generated. Starting storage...")

    # Store in ChromaDB
    collection = store_in_chroma(all_chunks, embeddings)

    print(f"Process complete. Collection contains {collection.count()} documents.")

    return collection


if __name__ == "__main__":
    main()


def store_embeddings_in_chroma(chunk_embeddings):
    vector_db = Chroma(
        collection_name='transcript_knowledge_base',
        embedding_function=GoogleGenerativeAIEmbeddings(),

    )
    for chunk in chunk_embeddings:
        vector_db.add_texts(chunk['text'], embeddings=chunk['embedding'])
    return vector_db


transcripts_embeddings = generate_embeddings(all_chunks)
