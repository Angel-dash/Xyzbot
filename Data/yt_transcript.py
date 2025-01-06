from youtube_transcript_api import YouTubeTranscriptApi
from get_video_link import video_links_main
import os
from datetime import datetime

transcripts = []


def save_transcript(video_id, transcript_text, folder_name="transcripts"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{video_id}_{timestamp}.txt"
    filepath = os.path.join(folder_name, filename)

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
    video_links_list, new_video_added, new_videos_link = video_links_main()
    video_transcripts = {}

    # First load existing transcripts
    transcripts_folder = "transcripts"
    if os.path.exists(transcripts_folder):
        existing_files = os.listdir(transcripts_folder)
        for file in existing_files:
            video_id = file.split("_")[0]
            with open(os.path.join(transcripts_folder, file), "r", encoding="utf-8") as f:
                transcript_text = f.read().splitlines()
            video_transcripts[video_id] = {
                'text': transcript_text,
                'file_path': os.path.join(transcripts_folder, file)
            }

    # Then fetch new transcripts if there are any
    if new_video_added:
        print("New videos has been added... Fetching transcript for new videos only")
        new_transcripts = fetch_yt_transcript(new_videos_link)
        # Merge new transcripts with existing ones
        video_transcripts.update(new_transcripts)

    return video_transcripts


if __name__ == '__main__':
    full_transcripts = all_video_transcript_pipeline()
    print("this is full transcripts of all the youtube videos")
    print(full_transcripts)
