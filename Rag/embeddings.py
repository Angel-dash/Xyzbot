from Rag.chunking import split_text_to_chunks
from tqdm import tqdm
import numpy as np
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer

all_chunks = split_text_to_chunks()


def get_existing_documents(collection):
    """Get all existing documents and their IDs from the collection"""
    if collection.count() == 0:
        return set(), {}

    # Get all documents from collection
    result = collection.get()
    existing_docs = {doc: id for doc, id in zip(result['documents'], result['ids'])}
    existing_ids = set(result['ids'])
    return existing_ids, existing_docs


def generate_embeddings(splits, batch_size=32):
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


def store_in_chroma(chunks, embeddings, update_existing=True):
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

    # Prepare new documents for insertion
    new_docs = []
    new_embeddings = []
    new_ids = []
    new_metadatas = []

    print("Checking for new documents...")
    for i, chunk in enumerate(chunks):
        doc_content = chunk.page_content
        # Only add documents that don't exist
        if doc_content not in existing_docs:
            new_docs.append(doc_content)
            new_embeddings.append(embeddings[i])
            new_ids.append(str(len(existing_ids) + len(new_docs) - 1))
            new_metadatas.append(chunk.metadata)

    if new_docs:
        print(f"Adding {len(new_docs)} new documents to collection...")
        # Add only new documents to collection
        collection.add(
            ids=new_ids,
            documents=new_docs,
            embeddings=new_embeddings,
            metadatas=new_metadatas
        )
        print(f"Added {len(new_docs)} new documents")
    else:
        print("No new documents to add")

    return collection


def main():
    all_chunks = split_text_to_chunks()

    print(f"Starting embedding generation for {len(all_chunks)} chunks...")
    embeddings = generate_embeddings(all_chunks)
    print("Embeddings generated. Starting storage...")
    collection = store_in_chroma(all_chunks, embeddings)

    print(f"Process complete. Collection contains {collection.count()} documents.")

    return collection


if __name__ == "__main__":
    main()
