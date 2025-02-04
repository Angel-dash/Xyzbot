Andrew Huberman RAG-Based AI Chatbot Using YouTube Videos
📌 Overview
Xyzbot is an AI chatbot designed to mimic Andrew Huberman by fetching and analyzing YouTube video transcripts from his channel. It automatically retrieves transcripts when new videos are uploaded, updates its knowledge base in ChromaDB, and provides citations linking to the specific video sources. The application is built using Streamlit and deployed in a Docker container.

🚀 Features
Mimics Andrew Huberman by extracting insights from his YouTube videos
Automatically fetches transcripts when new videos are uploaded
Stores and updates knowledge base using ChromaDB
Uses RAG to generate accurate, citation-linked responses
Provides direct links to cited YouTube videos
Interactive Streamlit UI for seamless user experience
Deployed using Docker for easy scalability and portability
🛠 Tech Stack
Backend: Python, LangChain, OpenAI API
Frontend: Streamlit
Database: ChromaDB (Vector Store)
Deployment: Docker
📂 Project Structure
markdown
Copy
📦 Xyzbot
├── 📂 Data
├── 📂 Example
│   ├── __init__.py
│   ├── rag_example.py
├── 📂 Llm
│   ├── __init__.py
│   ├── llm_endpoints.py
├── 📂 Notebook
├── 📂 Prompts
│   ├── __init__.py
│   ├── huberman_prompt.py
│   ├── summary_prompt.py
├── 📂 Rag
│   ├── chromadb.db
│   ├── 📂 Processed_folder
│   │   ├── __init__.py
│   │   ├── error_log.txt
│   ├── rag_pipeline.py
├── 📂 utils
│   ├── __init__.py
│   ├── corefrence.py
│   ├── get_link.py
│   ├── summarization.py
├── .dockerignore
├── .env
├── .gitignore
├── Dockerfile
├── poetry.lock
├── pyproject.toml
🔧 Installation
Clone the repository:
bash
Copy
git clone https://github.com/Angel-dash/Xyzbot.git
cd Xyzbot
Create a virtual environment and install dependencies:
bash
Copy
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
🎯 API Keys Required
To run this project, you'll need to get the following API keys:

Google Gemini API Key (For language processing and AI capabilities)
YouTube API Key (For fetching transcripts from YouTube)
How to Get Your API Keys:
Google Gemini API Key: Sign up for Google Cloud and generate an API key here.
YouTube API Key: Get your YouTube API key from the Google Developer Console.
Setting Up Your .env File
After obtaining the API keys, create a .env file in the root of the project (or use the provided .env template) and add the following lines:

bash
Copy
GOOGLE_API_KEY=your_google_gemini_api_key_here
API_KEY=your_youtube_api_key_here
Make sure to replace your_google_gemini_api_key_here and your_youtube_api_key_here with the actual keys you obtained.

▶️ Running the Application
Start the Streamlit app:
bash
Copy
streamlit run src/main.py
Open http://localhost:8501/ in your browser.
🐳 Running with Docker
Build the Docker image:
bash
Copy
docker build -t xyzbot:latest .
Run the container:
bash
Copy
docker run -p 8501:8501 xyzbot:latest
📌 Future Enhancements
Improve response generation with fine-tuned LLMs
Enable real-time monitoring of multiple YouTube channels
Enhance citation formatting for better user experience
Provide timestamps for specific content along with the links
AI agent ability to detect greetings and unrelated topics
Improve RAG by using a hybrid method
Implement caching for better performance
📜 License
This project is licensed under the MIT License.

🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.
