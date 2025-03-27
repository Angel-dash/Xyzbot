from youtube_transcript_api import YouTubeTranscriptApi
from Data.get_video_link import video_links_main
from pathlib import Path
from datetime import datetime
import os 

# Dynamically get the root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Moves up from /Data/
TRANSCRIPTS_FOLDER = Path(os.getenv("TRANSCRIPTS_FOLDER", str(PROJECT_ROOT / "Data" / "transcripts")))


def save_transcript(video_id, transcript_text):
    """
    Saves transcripts to the local folder
    """
    # Ensure the transcripts folder exists
    TRANSCRIPTS_FOLDER.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{video_id}_{timestamp}.txt"
    file_path = TRANSCRIPTS_FOLDER / filename

    file_path.write_text('\n'.join(transcript_text), encoding="utf-8")
    return file_path


def get_video_id(video_links_list):
    return [link.replace("https://www.youtube.com/watch?v=", "") for link in video_links_list]


def fetch_yt_transcript(video_ids):
    """
    Fetches YouTube transcripts using video IDs.
    """
    video_transcripts = {}

    for video_id in video_ids:
        print(f"Fetching transcript for: {video_id}")
        try:
            output = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = [item['text'] for item in output]

            # Save transcript and get file path
            file_path = save_transcript(video_id, transcript_text)
            video_transcripts[video_id] = {
                'text': transcript_text,
                'file_path': str(file_path)
            }
            print(f"Transcript saved to: {file_path}")

        except Exception as e:
            print(f"Transcript not found for video: {video_id}")
            video_transcripts[video_id] = {
                'text': [],
                'file_path': None
            }

    return video_transcripts


def all_video_transcript_pipeline():
    """
    Handles fetching and storing transcripts, checking for new videos.
    """
    print(f"Looking for transcripts in: {TRANSCRIPTS_FOLDER}")
    video_links_list, new_video_added, new_videos_link = video_links_main()
    video_transcripts = {}

    # Always load existing transcripts
    if TRANSCRIPTS_FOLDER.exists():
        existing_files = list(TRANSCRIPTS_FOLDER.glob("*.txt"))
        print(f"Found {len(existing_files)} transcript files.")

        for file in existing_files:
            video_id = file.stem.split("_")[0]  # Extract video ID
            try:
                transcript_text = file.read_text(encoding="utf-8").splitlines()
                video_transcripts[video_id] = {
                    'text': transcript_text,
                    'file_path': str(file)
                }
                print(f"Loaded transcript for video: {video_id}")
            except Exception as e:
                print(f"Error loading transcript {file.name}: {e}")
    else:
        print(f"Transcripts folder not found at: {TRANSCRIPTS_FOLDER}, creating it.")
        TRANSCRIPTS_FOLDER.mkdir(parents=True, exist_ok=True)

    # Fetch new transcripts if needed
    if new_video_added and new_videos_link:
        print("New videos detected... Fetching transcripts.")
        new_video_ids = [url.split("v=")[1] for url in new_videos_link]  # Extract video IDs
        new_transcripts = fetch_yt_transcript(new_video_ids)

    print(f"Total transcripts loaded: {len(video_transcripts)}")

