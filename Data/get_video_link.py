import os
import requests
from dotenv import load_dotenv
from new_video_added import get_new_video_url
from datetime import datetime
import json

load_dotenv()

api_key = os.getenv('API_KEY')
BASE_URL = "https://www.googleapis.com/youtube/v3"
channel = "https://www.youtube.com/@hubermanlab/videos"
new_video_added = False
video_links_folder_name = "videolinks"


def get_chanel_id(chanel_name):
    url = f"{BASE_URL}/search"
    params = {
        "part": "snippet",
        "q": chanel_name,
        "type": "channel",
        "key": api_key
    }
    response = requests.get(url, params)
    response_data = response.json()
    if "items" in response_data and len(response_data["items"]) > 0:
        return response_data["items"][0]["snippet"]["channelId"]
    else:
        return None


def get_video_links(channel_id):
    url = f"{BASE_URL}/search"
    video_links = []
    next_page_token = None

    while True:
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "maxResults": 50,
            "type": "video",
            "key": api_key,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(url, params=params)
        response_data = response.json()

        if "items" not in response_data:
            break

        for item in response_data["items"]:
            video_id = item["id"]["videoId"]
            video_links.append(f"https://www.youtube.com/watch?v={video_id}")

        next_page_token = response_data.get("nextPageToken")
        if not next_page_token:
            break

    return video_links


def save_video_links(video_links):
    if not os.path.exists(video_links_folder_name):
        os.makedirs(video_links_folder_name)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"video_links_{timestamp}.json"
    filepath = os.path.join(video_links_folder_name, filename)
    with open(filepath, 'w') as file:
        json.dump(video_links, file)
    print(f"{len(video_links)} The video links is saved successfully to {filename}")


def load_video_links():
    """
    Load the most recent video links file based on timestamp in the filename.
    """
    # List all files in the current directory
    if not os.path.exists(video_links_folder_name):
        print(f"{video_links_folder_name} does not exits")
    files = [f for f in os.listdir(video_links_folder_name) if f.startswith("video_links_") and f.endswith(".json")]

    if not files:
        print("No video links file found.")
        return []

    # Sort files by the timestamp in their names (descending)
    files.sort(key=lambda x: datetime.strptime(x[len("video_links_"):-len(".json")], "%Y%m%d%H%M%S"), reverse=True)

    # Load the most recent file
    latest_file = files[0]
    filepath = os.path.join(video_links_folder_name, latest_file)
    try:
        with open(filepath, 'r') as file:
            video_links = json.load(file)
            print(f"{len(video_links)} video links loaded successfully from {latest_file}.")
            return video_links
    except Exception as e:
        print(f"Error loading {latest_file}: {e}")
        return []


def video_links_main():
    video_links = load_video_links()
    if video_links:
        print(f"Using {len(video_links)} saved video links")
    else:
        channel_name = input("Enter the YouTube channel name: ")
        channel_id = get_chanel_id(channel_name)

        if channel_id:
            print(f"Fetching videos for channel: {channel_name} (ID: {channel_id})")
            video_links = get_video_links(channel_id)
            save_video_links(video_links)
        else:
            print("Failed to fetch video links")
    # for link in video_links:
    #     # print(link)
    new_video_url = get_new_video_url(channel)
    # new_video_url = new_video_url[:3]
    new_videos = [url for url in new_video_url if url not in video_links]

    if new_videos:
        print(f"{len(new_videos)} new video founds")
        video_links.extend(new_videos)
        save_video_links(video_links)
        new_video_added = True
    else:
        print("No new video founds")
        new_video_added = False
    # print(new_video_added)
    return video_links, new_video_added, new_videos


if __name__ == "__main__":
    video_links_main()
