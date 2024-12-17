import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_KEY')

def get_chanel_id(chanel_name):
  url = f"{BASE_URL}/search"
  params = {
      "part":"snippet",
      "q":chanel_name,
      "type":"channel",
      "key": API_KEY
  }
  response = requests.get(url,params)
  response_data = response.json()
  if "items" in response_data and len(response_data["items"]) > 0:
    return response_data["items"][0]["snippet"]["channelId"]
  else:
    return None