from Rag.chunking import split_text_to_chunks
from tqdm import tqdm
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import os
import shutil


def ensure_db_permissions():
    """Ensure the database directory has correct permissions"""
    db_path = os.path.abspath("db")

    # Remove existing db if it exists
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
            print("Removed existing database directory")
        except Exception as e:
            print(f"Error removing existing database: {str(e)}")
            raise

    # Create new directory with proper permissions
    try:
        os.makedirs(db_path, mode=0o777, exist_ok=True)
        print(f"Created database directory with full permissions at: {db_path}")
    except Exception as e:
        print(f"Error creating database directory: {str(e)}")
        raise

    return db_path


def store_in_chroma(chunks):
    # Ensure proper database permissions
    db_path = ensure_db_permissions()

    # Initialize Chroma client with new configuration
    client = chromadb.PersistentClient(path=db_path)

    try:
        # Create new collection
        collection = client.create_collection(
            name="transcript_collection",
            metadata={"description": "Video transcript embeddings"}
        )
        print("Created new collection")
    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        raise

    # Prepare documents for insertion
    docs = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]

    print(f"Generating embeddings for {len(docs)} documents...")

    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = []

    batch_size = 32
    with tqdm(total=len(docs), desc="Generating embeddings") as pbar:
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_embeddings = model.encode(batch)
            embeddings.extend(batch_embeddings)
            pbar.update(len(batch))

    print(f"Adding {len(docs)} documents to collection...")
    try:
        # Add documents to collection
        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Successfully added {len(docs)} documents")
    except Exception as e:
        print(f"Error adding documents: {str(e)}")
        raise

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