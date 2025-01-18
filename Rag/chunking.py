import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import os
import sys
import json
from Data.yt_transcript import all_video_transcript_pipeline
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
all_video_transcript_pipeline()

full_transcripts = "text"
chromadb_path = "/home/nightwing/Codes/Xyzbot/Rag/chromadb.db"
transcripts_folder_path = '/home/nightwing/Codes/Xyzbot/Data/transcripts'
processed_files_path = "/home/nightwing/Codes/Xyzbot/Rag/Processed_folder/processed_files.json"

client = chromadb.PersistentClient(path=chromadb_path)
collection = client.get_or_create_collection(name="yt_transcript_collection")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# loader = TextLoader(full_transcripts)
import logging


# logging.basicConfig(level=logging.INFO)
#
#
# def prepare_documents(full_transcript):
#     docs = []
#     for key, value in full_transcript.items():
#         if isinstance(value, dict) and "text" in value:
#             content = " ".join(value["text"]) if isinstance(value["text"], list) else value["text"]
#             docs.append(Document(page_content=content, metadata={"source": key}))
#     return docs
#
#
def split_text_to_chunks(docs):
    try:
        logging.info(f"{len(docs)} documents prepared")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=['\n\n', '.', '?', '!'])
        chunks = text_splitter.split_documents(docs)
        return chunks
    except Exception as e:
        logging.error(f"Error while splitting text: {str(e)}")
        # Optionally log the full traceback to a file
        import traceback
        with open("error_log.txt", "w") as f:
            traceback.print_exc(file=f)
        return None
#

def load_new_transcripts(transcripts_folder_path, processed_files):
    docs = []
    current_files = os.listdir(transcripts_folder_path)
    new_files = [f for f in current_files if f.endswith(".txt") and f not in processed_files]
    for file_name in new_files:
        file_path = os.path.join(transcripts_folder_path, file_name)
        with open(file_path, "r", encoding='utf-8') as f:
            content = f.read()
            docs.append(Document(page_content=content, metadata={'source':file_name}))
    return docs, new_files

def update_processed_files(file_path, new_files):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            processed_files = json.load(f)
    else:
        processed_files = []

    processed_files.extend(new_files)
    with open(file_path, "w") as f:
        json.dump(processed_files, f)

    return processed_files

if os.path.exists(processed_files_path):
    with open(processed_files_path, "r") as f:
        processed_files = json.load(f)
else:
    processed_files = []


new_docs, new_files = load_new_transcripts(transcripts_folder_path, processed_files)

if new_docs:
    # Split into chunks
    chunks = split_text_to_chunks(new_docs)
    if chunks:
        # Here, calculate embeddings and add to your vector database
        print(f"Added {len(new_files)} new files to the database.")
        for doc in new_docs:
            print(f"Processed file: {doc.metadata['source']}")
else:
    print("No new files to process.")

# Update the record of processed files
processed_files = update_processed_files(processed_files_path, new_files)
