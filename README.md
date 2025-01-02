Agentic AI using YouTube Transcript-API & RAG
Overview
This project focuses on creating an agentic AI chatbot that leverages YouTube videos as its knowledge base. By utilizing the YouTube Transcript API and Retrieval-Augmented Generation (RAG), the AI bot can extract information from YouTube video transcripts and answer queries based on that content.

The process of fetching new YouTube videos and extracting their transcripts is automated using GitHub Actions. This ensures that the knowledge base is continuously updated with new content from YouTube, which can be used by the bot for answering various questions.

Features
Automated Video Fetching: GitHub Actions automatically fetches new YouTube videos and updates the dataset regularly.
Transcript Extraction: The YouTube Transcript API is used to extract transcripts from the fetched YouTube videos.
Retrieval-Augmented Generation (RAG): Utilizes the RAG approach to query the AI and retrieve information from video transcripts to answer user queries.
Bot Interaction: A chatbot interface that answers questions based on the YouTube transcripts.
