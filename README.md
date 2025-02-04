# Andrew Huberman RAG-Based AI Chatbot Using YouTube Videos

## 📌 Overview
Xyzbot is an AI chatbot designed to mimic Andrew Huberman by fetching and analyzing YouTube video transcripts from his channel. It automatically retrieves transcripts when new videos are uploaded, updates its knowledge base in ChromaDB, and provides citations linking to the specific video sources. The application is built using Streamlit and deployed in a Docker container.

## 🚀 Features
- Mimics Andrew Huberman by extracting insights from his YouTube videos
- Automatically fetches transcripts when new videos are uploaded
- Stores and updates knowledge base using ChromaDB
- Uses RAG to generate accurate, citation-linked responses
- Provides direct links to cited YouTube videos
- Interactive Streamlit UI for seamless user experience
- Deployed using Docker for easy scalability and portability

## 🛠 Tech Stack
- **Backend**: Python, LangChain, OpenAI API
- **Frontend**: Streamlit
- **Database**: ChromaDB (Vector Store)
- **Deployment**: Docker

## 📂 Project Structure
```
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
```

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Angel-dash/Xyzbot.git
   cd Xyzbot
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## ▶️ Running the Application
1. Start the Streamlit app:
   ```bash
   streamlit run src/main.py
   ```
2. Open `http://localhost:8501/` in your browser.

## 🐳 Running with Docker
1. Build the Docker image:
   ```bash
   docker build -t xyzbot:latest .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 xyzbot:latest
   ```

## 📌 Future Enhancements
- Improve response generation with fine-tuned LLMs
- Enable real-time monitoring of multiple YouTube channels
- Enhance citation formatting for better user experience
- Provide timestamps for specific content along with the links
- AI agent ability to detect greetings and unrelated topics
- Improve RAG by using a hybrid method
- Implement caching for better performance

## 📜 License
This project is licensed under the MIT License.

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---
**Author:** Angel Dash | **GitHub:** [Angel-dash](https://github.com/Angel-dash)

