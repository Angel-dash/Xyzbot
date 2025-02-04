import sys
import chromadb
# transcripts_folder_path = '/home/nightwing/Codes/Xyzbot/Data/transcripts'
transcripts_folder_path = 'Data/transcripts'
chromadb_path = "Rag/chromadb.db"
client = chromadb.PersistentClient(path=chromadb_path)
collection = client.get_or_create_collection(name="yt_transcript_collection")

print("Python path:", sys.path)
from Rag.rag_pipeline import main_workflow

# Run the application
if __name__ == "__main__":
    main_workflow(transcripts_folder_path, collection)