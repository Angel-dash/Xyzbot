from Rag.chunking import split_text_to_chunks
from tqdm import tqdm
import numpy as np
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer


def get_existing_documents(collection):
    """Get all existing documents and their IDs from the collection"""
    if collection.count() == 0:
        return set(), {}

    # Get all documents from collection
    result = collection.get()
    existing_docs = {doc: id for doc, id in zip(result['documents'], result['ids'])}
    existing_ids = set(result['ids'])
    return existing_ids, existing_docs


def generate_embeddings_for_docs(documents, batch_size=32):
    """Generate embeddings only for specific documents"""
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    chunks_embeddings = []
    with tqdm(total=len(documents), desc="Generating embeddings") as pbar:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_embeddings = model.encode(batch)
            chunks_embeddings.extend(batch_embeddings)
            pbar.update(len(batch))

    return np.array(chunks_embeddings)


def store_in_chroma(chunks):
    # Initialize Chroma client
    client = chromadb.Client(Settings(
        persist_directory="db"
    ))

    try:
        # Try to get existing collection
        collection = client.get_collection(
            name="transcript_collection"
        )
        print("Found existing collection")
    except:
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name="transcript_collection",
            metadata={"description": "Video transcript embeddings"}
        )
        print("Created new collection")

    # Get existing documents
    existing_ids, existing_docs = get_existing_documents(collection)

    # Filter out chunks that already exist
    new_chunks = []
    for chunk in chunks:
        if chunk.page_content not in existing_docs:
            new_chunks.append(chunk)

    if not new_chunks:
        print("No new documents to add")
        return collection

    print(f"Found {len(new_chunks)} new documents. Generating embeddings...")

    # Prepare new documents for insertion
    new_docs = [chunk.page_content for chunk in new_chunks]
    new_metadatas = [chunk.metadata for chunk in new_chunks]
    new_ids = [str(len(existing_ids) + i) for i in range(len(new_chunks))]

    # Generate embeddings only for new documents
    new_embeddings = generate_embeddings_for_docs(new_docs)

    print(f"Adding {len(new_docs)} new documents to collection...")
    # Add only new documents to collection
    collection.add(
        ids=new_ids,
        documents=new_docs,
        embeddings=new_embeddings.tolist(),
        metadatas=new_metadatas
    )
    print(f"Added {len(new_docs)} new documents")

    return collection


def main():
    print("Starting chunking process...")
    chunks = split_text_to_chunks()
    print(f"Generated {len(chunks)} chunks")

    collection = store_in_chroma(chunks)
    print(f"Process complete. Collection contains {collection.count()} documents.")
    return collection


if __name__ == "__main__":
    main()