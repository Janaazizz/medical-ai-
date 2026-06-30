---
title: Medical Ai Bot
emoji: 💻
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
---

# Medical AI Chatbot (RAG-Based)

An advanced, context-aware AI medical assistant utilizing a Retrieval-Augmented Generation (RAG) pipeline to deliver precise responses from processed medical documents.

## 🚀 How to Run Locally

### Prerequisites:
Ensure you have the project repository cloned to your local machine and your command-line interface navigated inside the directory folder.

### Step 1: Initialize Your Environment
Create and activate an isolated Conda development environment:
```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```
###step 2 : install Dependencies
```bash
pip install -r requirements.txt
```
###step 3: env

PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

###Step 4: Run the Embedding Pipelines
```bash
python store_index.py
```
###Step 5: Launch the Application Server
```bash
python app.py
```
Once initialized successfully, open up your preferred internet browser and navigate to:

http://localhost:5000/

Tech Used:
Core Language: Python
AI Framework Orchestration: LangChain
Web App Framework: Flask
Large Language Model API: Google Gemini API
Vector Storage Database: Pinecone

 ###  [Click Here to Access the Live Hosted Web App](https://jaaziz1234-medical-ai-bot.hf.space)
Distributed cleanly under the official MIT License. See the license blocks inside the root path files for complete details.

Copyright (c) 2026 jana
