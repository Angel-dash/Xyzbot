from dotenv import load_dotenv
import os
import google.generativeai as genai


# Configure the Generative AI model with the API key from the environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# Function to get a response from the generative model
def get_llm_response(prompt: str) -> str:
    response = gemini_model.generate_content(prompt)
    return response.text
