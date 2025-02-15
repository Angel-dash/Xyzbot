import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from Llm.llm_endpoints import get_llm_response
from utils.get_link import get_source_link
from Prompts.huberman_prompt import huberman_prompt

# Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

chromadb_path = "app/Rag/chromadb.db"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# Helper Functions
def split_text_to_chunks(docs, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(docs)
    return chunks


def get_new_files(transcripts_folder_path, collection):
    """Find new transcript files that haven't been processed yet."""
    all_files = [f for f in os.listdir(transcripts_folder_path) if f.endswith(".txt")]
    existing_files = [meta["source"] for meta in collection.get()['metadatas']]
    return [f for f in all_files if f not in existing_files]


def process_single_file(file_path):
    """Process a single file and return its chunks."""
    with open(file_path, 'r') as f:
        content = f.read()
    chunks = split_text_to_chunks(content)
    return chunks, os.path.basename(file_path)


def batch_embed_chunks(chunks, batch_size=8):
    """Embed chunks in batches."""
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings.tolist())
    return embeddings


def process_and_add_new_files(transcripts_folder_path, collection):
    """Process and add new transcript files to the vector database."""
    new_files = get_new_files(transcripts_folder_path, collection)
    if not new_files:
        logging.info("No new files to process")
        return False

    # Use a reasonable number of workers (4 is usually a good default)
    n_workers = min(4, len(new_files))
    logging.info(f"Using {n_workers} workers for processing")

    all_chunks = []
    all_metadata = []
    all_ids = []

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_single_file, os.path.join(transcripts_folder_path, file)): file
            for file in new_files
        }

        for future in as_completed(futures):
            file = futures[future]
            try:
                chunks, filename = future.result()
                file_metadata = [{"source": filename} for _ in range(len(chunks))]
                file_ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

                all_chunks.extend(chunks)
                all_metadata.extend(file_metadata)
                all_ids.extend(file_ids)

                logging.info(f"Processed {filename}")
            except Exception as e:
                logging.error(f"Error processing {file}: {str(e)}")
                continue

    # Process embeddings in batches
    logging.info(f"Generating embeddings for {len(all_chunks)} chunks")
    embeddings = batch_embed_chunks(all_chunks)

    # Add to database in batches
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        end_idx = min(i + batch_size, len(all_chunks))
        collection.upsert(
            documents=all_chunks[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=all_metadata[i:end_idx],
            ids=all_ids[i:end_idx]
        )
        logging.info(f"Added batch {i // batch_size + 1} to database")

    logging.info(f"Successfully processed {len(new_files)} files")
    return True


def query_database(collection, query_text, n_results=3):
    """Retrieve the most relevant chunks for the query."""
    query_embeddings = embedding_model.encode(query_text).tolist()
    results = collection.query(query_embeddings=query_embeddings, n_results=n_results)
    retrieved_docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    return retrieved_docs, metadatas


def enhance_query_with_history(query_text, summarized_history):
    enhance_query = f"{query_text}*2\n\n{summarized_history}"
    return enhance_query


def update_conversation_history(history, user_query, bot_response):
    """Update and keeps track of conversation history between user and the bot."""
    history.append({"user": user_query, "bot": bot_response})
    return history


def generate_response(conversation_history, query_text, retrieved_docs, source_links):
    """Generate a response using retrieved documents and the generative AI model."""
    context = " ".join(retrieved_docs)
    history_str = "\n".join([f"User: {turn['user']}\nBot: {turn['bot']}" for turn in conversation_history])
    sources_str = "\n".join(source_links)

    prompt = huberman_prompt.format(
        context=context,
        sources=sources_str,
        history=history_str,
        question=query_text
    )

    response = get_llm_response(prompt)
    full_response = f"{response}\n\nSources:\n{sources_str}"
    return full_response


def main_workflow(transcripts_folder_path, collection):
    """Run the full RAG workflow."""
    new_files_added = process_and_add_new_files(transcripts_folder_path, collection)
    if new_files_added:
        logging.info("New transcripts added to the database.")
    else:
        logging.info("No new files found. Using existing database.")

    conversation_history = []

    while True:
        query_text = input("\nEnter your query(or type 'exit' to end):").strip()
        if query_text.lower() == "exit":
            print("Ending the conversation. Goodbye")
            break

        query_text_with_conversation_history = enhance_query_with_history(query_text, conversation_history)
        retrived_docs, metadatas = query_database(collection, query_text_with_conversation_history)
        print("-" * 50)
        source_link = get_source_link(metadatas)
        print(source_link)
        print("-" * 50)

        if not retrived_docs:
            print("No relevent documents is found")
            continue

        response = generate_response(conversation_history, query_text, retrived_docs, source_link)
        conversation_history = update_conversation_history(conversation_history, query_text, response)
        print("\nGenerated Response:")
        print(response)