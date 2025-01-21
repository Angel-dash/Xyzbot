from Rag.chunking import split_text_to_chunks
from tqdm import tqdm
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import os
import shutil


def get_embeddings(docs, batch_size=32):
    """Generate embeddings for documents using sentence transformer"""
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = []

    with tqdm(total=len(docs), desc="Generating embeddings") as pbar:
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_embeddings = model.encode(batch)
            embeddings.extend(batch_embeddings)
            pbar.update(len(batch))

    return embeddings


def initialize_chroma_client(db_path):
    """Initialize ChromaDB client without removing existing database"""
    if not os.path.exists(db_path):
        try:
            os.makedirs(db_path, mode=0o777, exist_ok=True)
            print(f"Created database directory at: {db_path}")
        except Exception as e:
            print(f"Error creating database directory: {str(e)}")
            raise

    return chromadb.PersistentClient(path=db_path)


def get_or_create_collection(client, collection_name="transcript_collection"):
    """Get existing collection or create new one if it doesn't exist"""
    try:
        # Try to get existing collection
        collection = client.get_collection(name=collection_name)
        print(f"Found existing collection with {collection.count()} documents")
        return collection
    except Exception:
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Video transcript embeddings"}
        )
        print("Created new collection")
        return collection


def process_new_chunks(chunks, collection):
    """Process and add new chunks to the collection"""
    # Prepare documents for insertion
    docs = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Generate new IDs starting after existing documents
    start_id = collection.count()
    ids = [str(i) for i in range(start_id, start_id + len(chunks))]

    print(f"Generating embeddings for {len(docs)} new documents...")
    embeddings = get_embeddings(docs)

    print(f"Adding {len(docs)} new documents to collection...")
    try:
        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Successfully added {len(docs)} new documents")
    except Exception as e:
        print(f"Error adding documents: {str(e)}")
        raise


def store_in_chroma(chunks, db_path="db"):
    """Store chunks in ChromaDB, handling both new and existing databases"""
    # Initialize client without removing existing DB
    client = initialize_chroma_client(db_path)

    # Get existing collection or create new one
    collection = get_or_create_collection(client)

    # Process and add new chunks
    process_new_chunks(chunks, collection)

    return collection


def main():
    print("Starting chunking process...")
    chunks = split_text_to_chunks()
    print(f"Generated {len(chunks)} chunks")

    try:
        collection = store_in_chroma(chunks)
        final_count = collection.count()
        print(f"Process complete. Collection contains {final_count} documents.")
        return collection
    except Exception as e:
        print(f"Process failed: {str(e)}")
        return None


if __name__ == "__main__":
    main()

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name = "my_collection")
collection.upsert(
    documents = [
        "This is a dcouments about pineapple",
        "this is a document about oranges"
    ],
    ids = ['id1', 'id2']

)
results = collection.query(
    query_texts=["This is a query document about florida"], # Chroma will embed this for you
    n_results=2 # how many results to return
)

print(results)