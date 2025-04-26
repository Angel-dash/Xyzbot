import sys
import chromadb
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Rag"))
transcripts_folder_path_str = str(PROJECT_ROOT / "Data" / "transcripts")
chromadb_path_str = str(PROJECT_ROOT / "Rag" / "chromadb.db")

collection_name = "yt_transcript_collection"

from Rag.rag_pipeline import run_chat_loop,setup_rag_pipeline

if __name__ == "__main__":
    compiled_app, collection, embedding_model = setup_rag_pipeline(
        chromadb_path=chromadb_path_str,
        transcripts_folder_path=transcripts_folder_path_str,
        collection_name=collection_name
    )
    if compiled_app and collection:
        run_chat_loop(compiled_app, collection)
    else:
        print("\nApplication failed during setup. Exiting.")