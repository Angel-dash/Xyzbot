import requests
import re



def get_new_video_url(channel):
    """
    Fetch all video URLs from the given YouTube channel page.
    """
    try:
        html = requests.get(channel).text
        # Extract all video IDs from the HTML
        video_ids = re.findall(r'(?<="videoId":").*?(?=")', html)
        video_urls = [f"https://www.youtube.com/watch?v={video_id}" for video_id in video_ids]

        # Remove duplicates while preserving order
        video_urls = list(dict.fromkeys(video_urls))
        print(f"Fetched {len(video_urls)} video URLs from the channel.")
        return video_urls
    except Exception as e:
        print(f"Error fetching video URLs: {e}")
        return []
