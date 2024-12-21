from youtube_transcript_api import YouTubeTranscriptApi

from get_video_link import video_links_main

transcripts = []


def get_video_id(video_links_list):
    video_ids = []
    for links in video_links_list:
        video_id = links.replace("https://www.youtube.com/watch?v=", "")
        video_ids.append(video_id)

    return video_ids


def fetch_yt_transcript(video_ids):
    for id in video_ids:
        print(f"Fetching transcript for: {id}")
        try:
            output = YouTubeTranscriptApi.get_transcript(id)
            for transcript in output:
                transcripts.append(transcript['text'])
            print(f"Finished fetching transcript for: {id}")
        except Exception as e:
            print(f"Transcript not found for video: {id}")
            transcripts.append("")
    return transcripts


def all_video_transcript_pipeline():
    video_links_list = video_links_main()
    video_ids = get_video_id(video_links_list)
    full_transcripts = fetch_yt_transcript(video_ids)
    return full_transcripts


if __name__ == '__main__':
    full_transcripts = all_video_transcript_pipeline()
    print("this is full trnscirpts of all the youtube vides")
    print(full_transcripts)
