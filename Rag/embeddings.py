from Rag.chunking import split_text_to_chunks
from tqdm import tqdm
import numpy as np
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer
import os


def get_existing_documents(collection):
    """Get all existing documents and their IDs from the collection"""
    try:
        count = collection.count()
        print(f"Found {count} existing documents in collection")

        if count == 0:
            return set(), {}

        # Get all documents from collection
        result = collection.get(include=['documents', 'ids'])
        print(f"Successfully retrieved {len(result['documents'])} documents from collection")

        existing_docs = {doc: id for doc, id in zip(result['documents'], result['ids'])}
        existing_ids = set(result['ids'])
        return existing_ids, existing_docs
    except Exception as e:
        print(f"Error getting existing documents: {str(e)}")
        return set(), {}


def store_in_chroma(chunks):
    # Ensure the db directory exists
    db_path = os.path.abspath("db")
    os.makedirs(db_path, exist_ok=True)

    print(f"Using database path: {db_path}")

    # Initialize Chroma client with persistent storage
    client = chromadb.PersistentClient(
        path=db_path,
    )
    try:
        # Try to get existing collection
        collection = client.get_collection(name="transcript_collection")
        print("Found existing collection")
    except Exception as e:
        print(f"No existing collection found ({str(e)}), creating new one...")
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name="transcript_collection",
            metadata={"description": "Video transcript embeddings"}
        )
        print("Created new collection")

    # Get existing documents
    existing_ids, existing_docs = get_existing_documents(collection)
    print(f"Retrieved {len(existing_docs)} existing documents")

    # Filter out chunks that already exist
    new_chunks = []
    for chunk in chunks:
        if chunk.page_content not in existing_docs:
            new_chunks.append(chunk)

    print(f"Found {len(new_chunks)} new chunks out of {len(chunks)} total chunks")

    if not new_chunks:
        print("No new documents to add")
        return collection

    print(f"Generating embeddings for {len(new_chunks)} new documents...")

    # Prepare new documents for insertion
    new_docs = [chunk.page_content for chunk in new_chunks]
    new_metadatas = [chunk.metadata for chunk in new_chunks]
    new_ids = [str(len(existing_ids) + i) for i in range(len(new_chunks))]

    # Generate embeddings only for new documents
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    new_embeddings = []

    batch_size = 32
    with tqdm(total=len(new_docs), desc="Generating embeddings") as pbar:
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i:i + batch_size]
            batch_embeddings = model.encode(batch)
            new_embeddings.extend(batch_embeddings)
            pbar.update(len(batch))

    print(f"Adding {len(new_docs)} new documents to collection...")
    try:
        # Add only new documents to collection
        collection.add(
            ids=new_ids,
            documents=new_docs,
            embeddings=new_embeddings,
            metadatas=new_metadatas
        )
        # Explicitly persist the changes
        client.persist()
        print(f"Successfully added {len(new_docs)} new documents")
    except Exception as e:
        print(f"Error adding documents: {str(e)}")

    return collection


def main():
    print("Starting chunking process...")
    chunks = split_text_to_chunks()
    print(f"Generated {len(chunks)} chunks")

    collection = store_in_chroma(chunks)
    final_count = collection.count()
    print(f"Process complete. Collection contains {final_count} documents.")
    return collection


if __name__ == "__main__":
    main()