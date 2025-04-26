import gradio as gr
import chromadb
from typing import List, Dict
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

import os 
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "Rag"))
sys.path.append(str(project_root / "Data"))
sys.path.append(str(project_root / "Data" / "transcripts"))
sys.path.append(str(project_root / "Data" / "video_links"))
sys.path.append(str(project_root / "Llm"))
sys.path.append(str(project_root / "Prompts"))
sys.path.append(str(project_root / "utils"))
from Rag.rag_pipeline import (
    query_database,
    generate_response,
    enhance_query_with_history,
    update_conversation_history,
    process_and_add_new_files
)
from Data.yt_transcript import TRANSCRIPTS_FOLDER

INTRODUCTION = """
# 🧠 Welcome to HubermanBot!

I am your AI assistant trained on Andrew Huberman's podcast content. My knowledge base includes detailed information about:

- 🎯 Peak Performance & Focus
- 😴 Sleep Science & Optimization
- 🏋️ Physical Fitness & Recovery
- 🧘 Mental Health & Stress Management
- 🧪 Neuroscience & Biology
- 💪 Habit Formation & Behavior Change

For each response, I'll provide:
- Detailed answers based on podcast content
- Direct source links to specific episodes
- Scientific context when available

Ask me anything about these topics, and I'll help you find relevant information from the Huberman Lab Podcast!

Example questions you might ask:
- "What does Dr. Huberman recommend for better sleep?"
- "How can I improve my focus and concentration?"
- "What are the best practices for morning routines?"
"""


def initialize_chroma_client(rag_path: Path):
    print(f"Initializing ChromaDB at: {rag_path}")
    client = chromadb.PersistentClient(path=str(rag_path))
    print(f"Available collections: {client.list_collections()}")
    try:
        collection = client.get_collection(name="yt_transcript_collection")
        print(f"Found existing collection with {len(collection.get()['ids'])} documents")
    except Exception as e:
        print(f"No existing collection found, creating new one: {str(e)}")
        collection = client.create_collection(name="yt_transcript_collection")
    return collection


def format_youtube_url(filename: str) -> str:
    """Convert filename to YouTube URL"""
    video_id = filename.split('_')[0]
    return f"https://www.youtube.com/watch?v={video_id}"


class RAGChatInterface:
    def __init__(self, transcripts_folder_path: str, collection):
        self.transcripts_folder_path = transcripts_folder_path
        self.collection = collection
        self.conversation_history: List[Dict[str, str]] = []

    def process_query(self, message: str, history: List[List[str]]) -> str:
        """Process a single query and return the response"""
        self.conversation_history = [
            {"user": user_msg, "bot": bot_msg}
            for user_msg, bot_msg in history
        ]

        query_with_history = enhance_query_with_history(message, self.conversation_history)
        retrieved_docs, metadatas = query_database(self.collection, query_with_history)

        if not retrieved_docs:
            return "I apologize, but I couldn't find any relevant information about that in my knowledge base. Could you try rephrasing your question or ask about a different topic covered in the Huberman Lab Podcast?"

        source_links = [meta["source"] for meta in metadatas]
        response = generate_response(
            self.conversation_history,
            message,
            retrieved_docs,
            source_links
        )

        unique_sources = list(set(source_links))
        youtube_urls = [format_youtube_url(source) for source in unique_sources]

        formatted_response = f"{response}\n\n---\n📚 **Source Episodes:**\n"
        for url in youtube_urls:
            formatted_response += f"- {url}\n"

        return formatted_response


def create_interface(transcripts_folder_path: str, collection) -> gr.Interface:
    """Create and configure the Gradio interface"""
    rag_chat = RAGChatInterface(transcripts_folder_path, collection)

    interface = gr.ChatInterface(
        fn=rag_chat.process_query,
        title="🧠 HubermanBot - Your Neuroscience & Wellness AI Assistant",
        description=INTRODUCTION,
        examples=[
            "What are Dr. Huberman's top recommendations for better sleep?",
            "How does sunlight exposure affect our circadian rhythm?",
            "What supplements does Dr. Huberman recommend for focus?",
            "What are the best practices for morning routines according to Dr. Huberman?",
            "How can I optimize my workout recovery based on neuroscience?",
        ],
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        )
    )

    return interface


def main():
    # Get paths using pathlib
    project_root = Path(__file__).parent
    # rag_path = project_root / "Rag" / "chromadb.db"
    rag_path = os.getenv("CHROMA_PERSISTENCE_DIRECTORY",str(project_root / "Rag" / "chromadb.db"))
    transcripts_folder_path = TRANSCRIPTS_FOLDER

    # Initialize ChromaDB with proper error handling
    print("Starting ChromaDB initialization...")
    collection = initialize_chroma_client(rag_path)
    print("ChromaDB initialization complete")

    # Process any new files
    print("Checking for new files...")
    new_files_added = process_and_add_new_files(str(transcripts_folder_path), collection)
    if not new_files_added:
        print("No new files to process")

    # Create and launch the interface
    print("Launching Gradio interface...")
    interface = create_interface(str(transcripts_folder_path), collection)
    interface.launch(share=True, server_port=7860)


if __name__ == "__main__":
    main()