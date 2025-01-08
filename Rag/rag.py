from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
import os
from typing import Dict, List
import os
import sys
from Data.yt_transcript import all_video_transcript_pipeline
import google.generativeai as genai

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
print("THIS IS PROJECT ROOT")
print(PROJECT_ROOT)
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
print(API_KEY)

full_transcripts = all_video_transcript_pipeline()
print("this is full transcripts of all the youtube videos")
print(full_transcripts)

# loader = TextLoader()