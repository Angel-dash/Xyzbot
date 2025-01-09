from youtube_transcript_api import YouTubeTranscriptApi
# from get_video_link import video_links_main
from Data.get_video_link import video_links_main
import os
from datetime import datetime

transcripts = []

import os
from datetime import datetime


def save_transcript(video_id, transcript_text, folder_name="Data/transcripts"):
   #using abosule path
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
    folder_path = os.path.join(base_dir, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{video_id}_{timestamp}.txt"
    filepath = os.path.join(folder_path, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write('\n'.join(transcript_text))

    return filepath


def get_video_id(video_links_list):
    video_ids = []
    for links in video_links_list:
        video_id = links.replace("https://www.youtube.com/watch?v=", "")
        video_ids.append(video_id)

    return video_ids


def fetch_yt_transcript(video_ids):
    video_transcripts = {}  # Dictionary to store transcripts for each video

    for video_id in video_ids:
        print(f"Fetching transcript for: {video_id}")
        try:
            output = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = [item['text'] for item in output]

            # Save transcript and get file path
            file_path = save_transcript(video_id, transcript_text)
            video_transcripts[video_id] = {
                'text': transcript_text,
                'file_path': file_path
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
    # Get the Data directory path
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    transcripts_folder = os.path.join(CURRENT_DIR, "transcripts")

    print(f"Looking for transcripts in: {transcripts_folder}")
    video_links_list, new_video_added, new_videos_link = video_links_main()
    video_transcripts = {}

    # Always load existing transcripts
    if os.path.exists(transcripts_folder):
        existing_files = os.listdir(transcripts_folder)
        print(f"Found {len(existing_files)} files in transcripts folder")

        for file in existing_files:
            if file.endswith('.txt'):  # Make sure we only process text files
                video_id = file.split("_")[0]
                file_path = os.path.join(transcripts_folder, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        transcript_text = f.read().splitlines()
                    video_transcripts[video_id] = {
                        'text': transcript_text,
                        'file_path': file_path
                    }
                    print(f"Loaded transcript for video: {video_id}")
                except Exception as e:
                    print(f"Error loading transcript {file}: {e}")
    else:
        print(f"Transcripts folder not found at: {transcripts_folder}")
        os.makedirs(transcripts_folder)
        print(f"Created transcripts folder at: {transcripts_folder}")

    # Then fetch new transcripts if there are any
    if new_video_added and new_videos_link:
        print("New videos have been added... Fetching transcripts for new videos")
        new_video_ids = [url.split("v=")[1] for url in new_videos_link]  # Extract video IDs
        new_transcripts = fetch_yt_transcript(new_video_ids)
        # Merge new transcripts with existing ones
        video_transcripts.update(new_transcripts)
        print(f"Added {len(new_transcripts)} new transcripts")

    print(f"Total transcripts loaded: {len(video_transcripts)}")
    return video_transcripts


# if __name__ == '__main__':
#     full_transcripts = all_video_transcript_pipeline()
#     print("this is full transcripts of all the youtube videos")
#     print(full_transcripts)
