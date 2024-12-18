import requests
import re


def get_new_video_url(channel):
    html = requests.get(channel).text
    info = re.search('(?<={"label":").*?(?="})', html).group()
    url = "https://www.youtube.com/watch?v=" + re.search('(?<="videoId":").*?(?=")', html).group()

    # print(info)
    print(url)
    return url
