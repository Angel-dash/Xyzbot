[# Agentic AI using YouTube Transcript-API & RAG

## Overview

This project focuses on creating an **agentic AI chatbot** that leverages **YouTube videos** as its knowledge base. By utilizing the **YouTube Transcript API** and **Retrieval-Augmented Generation (RAG)**, the AI bot can extract information from YouTube video transcripts and answer queries based on that content.

The process of fetching new YouTube videos and extracting their transcripts is automated using **GitHub Actions**, ensuring that the knowledge base is continuously updated with fresh content from YouTube.

## Features

- **Automated Video Fetching**: GitHub Actions automatically fetches new YouTube videos and updates the dataset regularly.
- **Transcript Extraction**: The **YouTube Transcript API** extracts transcripts from the fetched YouTube videos.
- **Retrieval-Augmented Generation (RAG)**: The bot uses RAG to query the AI and retrieve information from video transcripts to answer user queries.
- **Bot Interaction**: A chatbot interface answers questions based on the YouTube video transcripts.


## Project Structure

```bash
/data
    /scripts
        - fetch_new_videos.py   # Script to check for new YouTube videos
        - fetch_transcripts.py  # Script to fetch transcripts for new videos
    /transcripts
        - transcripts.csv      # CSV or JSON file where transcripts of videos are stored
/github-actions
    - fetch_and_update.yml    # GitHub Action for fetching new videos and updating transcripts
](https://github.com/Angel-dash/Xyzbot)
