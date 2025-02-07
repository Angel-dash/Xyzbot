# Andrew Huberman RAG-Based AI Chatbot

## Overview
Xyzbot is an AI chatbot that extracts and synthesizes insights from Andrew Huberman's YouTube videos. It automatically retrieves video transcripts, updates its knowledge base in ChromaDB, and provides citation-linked responses.

## 🚀 Key Features
- Mimics Andrew Huberman's insights using YouTube video transcripts
- Automatic transcript retrieval and knowledge base updates
- RAG-powered response generation with direct video citations
- Interactive Streamlit user interface
- Docker-based deployment for easy scalability

## 🛠 Tech Stack
- Backend: Python, LangChain, OpenAI API
- Frontend: Streamlit
- Database: ChromaDB
- Deployment: Docker

## 📂 Project Structure
```
📦 Xyzbot
├── 📂 Data
├── 📂 Example
├── 📂 Llm
├── 📂 Notebook
├── 📂 Prompts
├── 📂 Rag
│   ├── chromadb.db
│   └── 📂 Processed_folder
├── 📂 utils
├── Dockerfile
└── pyproject.toml
```

## 🔧 Prerequisites
- Python 3.8+
- Docker (optional)

## 🔑 API Keys Required
1. Google Gemini API Key
2. YouTube API Key

## 🚀 Installation

### Local Setup
1. Clone the repository
   ```bash
   git clone https://github.com/Angel-dash/Xyzbot.git
   cd Xyzbot
   ```

2. Create virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Docker Setup

#### Option 1: Build Locally
```bash
docker build -t xyzbot:v1.0 .
docker run -it \
  -v $(pwd)/Rag:/app/Rag:rw \
  -e GOOGLE_API_KEY=your_api_key \
  xyzbot:v1.0
```

#### Option 2: Pull from Docker Hub
```bash
docker pull angeldash/xyzbot:v1.0
docker run -it \
  -v $(pwd)/Rag:/app/Rag:rw \
  -e GOOGLE_API_KEY=your_api_key \
  angeldash/xyzbot:v1.0
```

## 🖥️ Running the Application
```bash
streamlit run src/main.py
```

## 📈 Future Roadmap
- Fine-tuned LLM response generation
- Real-time multi-channel monitoring
- Enhanced citation formatting
- AI agent conversation handling
- Performance optimization

## 📜 License
MIT License

## 🤝 Contributing
Contributions are welcome! Open an issue or submit a pull request.

---
**Author:** Angel Dash | **GitHub:** [@Angel-dash](https://github.com/Angel-dash)
