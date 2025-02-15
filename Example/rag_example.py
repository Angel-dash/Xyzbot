import sys
import chromadb
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# transcripts_folder_path = '/home/nightwing/Codes/Xyzbot/Data/transcripts'
# transcripts_folder_path = 'Data/transcripts'
transcripts_folder_path = PROJECT_ROOT / "Data" / "transcripts"
chromadb_path = PROJECT_ROOT / "Rag" / "chromadb.db"
client = chromadb.PersistentClient(path=str(chromadb_path))
collection = client.get_or_create_collection(name="yt_transcript_collection")
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Rag"))
# print("Python path:", sys.path)
from Rag.rag_pipeline import main_workflow

# Run the application
if __name__ == "__main__":
    main_workflow(transcripts_folder_path, collection)