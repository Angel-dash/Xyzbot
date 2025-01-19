import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import os
import sys
from Data.yt_transcript import all_video_transcript_pipeline
import google.generativeai as genai

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
full_transcripts = all_video_transcript_pipeline()
loader = TextLoader(full_transcripts)

import logging

logging.basicConfig(level=logging.INFO)


def prepare_documents(full_transcript):
    docs = []
    for key, value in full_transcript.items():
        if isinstance(value, dict) and "text" in value:
            content = " ".join(value["text"]) if isinstance(value["text"], list) else value["text"]
            docs.append(Document(page_content=content, metadata={"source": key}))
    return docs


def split_text_to_chunks():
    try:
        docs = prepare_documents(full_transcripts)
        logging.info(f"{len(docs)} documents prepared")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=['\n\n', '.', '?', '!'])
        splits = text_splitter.split_documents(docs)
        return splits
    except Exception as e:
        logging.error(f"Error while splitting text: {str(e)}")
        # Optionally log the full traceback to a file
        import traceback
        with open("error_log.txt", "w") as f:
            traceback.print_exc(file=f)
        return None


all_splits = split_text_to_chunks()
if all_splits:
    print(f"Total chunks created: {len(all_splits)}")
    print(all_splits[0].metadata)
    print(all_splits[1])
else:
    print("Splitting failed. Check logs for details.")
