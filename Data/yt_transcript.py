from youtube_transcript_api import YouTubeTranscriptApi

from get_video_link import video_links_main

video_links_list = video_links_main()


# print("This is video links list")
# print("*************************************")
# print(video_links_list)

def get_video_id(video_links_list):
    video_ids = []
    for links in video_links_list:
        video_id = links.replace("https://www.youtube.com/watch?v=", "")
        video_ids.append(video_id)

    return video_ids


video_ids = get_video_id(video_links_list)
print("This is the list of video ids")
print(video_ids)
