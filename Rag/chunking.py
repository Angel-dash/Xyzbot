import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import json
import logging
from Llm.llm_endpoints import get_llm_response
# Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

chromadb_path = "/home/nightwing/Codes/Xyzbot/Rag/chromadb.db"
transcripts_folder_path = '/home/nightwing/Codes/Xyzbot/Data/transcripts'
processed_files_path = "/home/nightwing/Codes/Xyzbot/Rag/Processed_folder/processed_files.json"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path=chromadb_path)
collection = client.get_or_create_collection(name="yt_transcript_collection")

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


def process_and_add_new_files(transcripts_folder_path, collection):
    """Process and add new transcript files to the vector database."""
    new_files = get_new_files(transcripts_folder_path, collection)
    if not new_files:
        return False

    for new_file in new_files:
        file_path = os.path.join(transcripts_folder_path, new_file)
        with open(file_path, 'r') as f:
            content = f.read()

        chunks = split_text_to_chunks(content)
        embeddings = embedding_model.encode(chunks).tolist()

        ids = [f"{new_file}_chunk_{i}" for i in range(len(chunks))]
        metadata = [{"source": new_file} for _ in range(len(chunks))]
        collection.upsert(documents=chunks, embeddings=embeddings, metadatas=metadata, ids=ids)

        logging.info(f"Added {new_file} to the database")
    return True


def query_database(collection, query_text, n_results=3):
    """Retrieve the most relevant chunks for the query."""
    query_embeddings = embedding_model.encode(query_text).tolist()
    results = collection.query(query_embeddings=query_embeddings, n_results=n_results)
    retrieved_docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    return retrieved_docs, metadatas

def update_conversation_history(history, user_query, bot_response):
    """
    Update and keeps track of conversation history between user and the bot
    :param history:
    :param user_query:
    :param bot_response:
    :return:
    """
    history.append({"user":user_query, "bot":bot_response})
    return history


def generate_response(conversation_history,query_text, retrieved_docs):
    """Generate a response using retrieved documents and the generative AI model."""

    context = " ".join(retrieved_docs)
    history_str = "\n".join([f"User: {turn['user']}\nBot: {turn['bot']}" for turn in conversation_history])
    prompt = f"Using the context below, answer the question:\n\nContext:\n{context}\n\nQuestion: {query_text}"
    response = get_llm_response(prompt)
    return response


# Main Workflow
def main_workflow(transcripts_folder_path, collection):
    """Run the full RAG workflow."""
    # Process new files
    new_files_added = process_and_add_new_files(transcripts_folder_path, collection)
    if new_files_added:
        logging.info("New transcripts added to the database.")
    else:
        logging.info("No new files found. Using existing database.")

    #Initialize conversation history
    conversation_history = []

    while True:
        query_text = input("\nEnter your query(or type 'exit' to end):")
        if query_text.lower() == "exit":
            print("Ending the conversation. Goodbye")
            break

        retrived_docs, metadatas = query_database(collection, query_text)
        print("-"*50)
        print(metadatas)
        print("-"*50)
        if not retrived_docs:
            print("No relevent documents is found")
            continue
        response = generate_response(conversation_history,query_text,retrived_docs)
        conversation_history = update_conversation_history(conversation_history,query_text,response)
        print("\nGenerated Response:")
        print(response)


# Run the application
if __name__ == "__main__":
    main_workflow(transcripts_folder_path, collection)
