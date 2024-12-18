import requests
import re

channel = "https://www.youtube.com/@hubermanlab/videos"

html = requests.get(channel).text
info = re.search('(?<={"label":").*?(?="})', html).group()
url = "https://www.youtube.com/watch?v=" + re.search('(?<="videoId":").*?(?=")', html).group()

# print(info)
print(url)