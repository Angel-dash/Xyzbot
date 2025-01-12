# Agentic AI using YouTube Transcript-API & RAG

## Overview

This project focuses on creating an **agentic AI chatbot** that leverages **YouTube videos** as its knowledge base. By utilizing the **YouTube Transcript API** and **Retrieval-Augmented Generation (RAG)**, the AI bot can extract information from YouTube video transcripts and answer queries based on that content.

The process of fetching new YouTube videos and extracting their transcripts is automated using **GitHub Actions**, ensuring that the knowledge base is continuously updated with fresh content from YouTube.

## Features

- **Automated Video Fetching**: GitHub Actions automatically fetches new YouTube videos and updates the dataset regularly.
- **Transcript Extraction**: The **YouTube Transcript API** extracts transcripts from the fetched YouTube videos.
- **Retrieval-Augmented Generation (RAG)**: The bot uses RAG to query the AI and retrieve information from video transcripts to answer user queries.
- **Bot Interaction**: A chatbot interface answers questions based on the YouTube video transcripts.

## How vector DB works 
First Check for Vector DB:
Tries to get existing collection named "transcript_collection"
If not found, creates a new one
If found, uses the existing one
Document Comparison:
Gets all existing documents from the database
Takes your new text chunks
Compares them to find which chunks are new (not in database)
Processing New Content:
If no new content is found → stops (nothing to do)
If new content exists → only generates embeddings for these new chunks
Update Database:
Takes the new embeddings
Adds them to the existing vector database
Maintains all previous data while adding new content
So if you have:

Original DB with chunks A, B, C
New text with chunks A, B, C, D, E
It will only process and add D and E to the database
This makes the process much more efficient since you're not reprocessing content that's already in the database!


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
