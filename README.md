# RAG-based AI Teaching Assistant

A retrieval-augmented generation (RAG) project designed to transform recorded web development lectures into an intelligent teaching assistant. The system processes lecture videos, converts them into text, creates semantic embeddings, and answers student questions using the most relevant course content.

## Project Overview

This project demonstrates how AI can be used to make educational content more accessible and interactive. Instead of manually searching through long lecture videos, students can ask natural-language questions and receive answers grounded in the course material.

## Key Features

- Converts lecture videos into audio and transcribed text
- Splits transcripts into meaningful chunks for retrieval
- Generates embeddings for semantic similarity search
- Retrieves the most relevant lesson segments for user queries
- Uses an LLM to generate course-aware answers
- Supports question answering for educational content with timestamp-based guidance

## Workflow

1. Place lecture videos in the `videos/` directory.
2. Convert videos to MP3 audio.
3. Transcribe the audio using Whisper.
4. Create text chunks and embeddings.
5. Retrieve relevant chunks for a user question.
6. Generate an answer grounded in the retrieved course content.

## Project Structure

- `process_video.py` – converts videos from `videos/` into MP3 files in `audios/`
- `mp3_to_json.py` – transcribes audio into JSON chunks using faster-whisper
- `preprocess_json.py` – generates embeddings and stores them in `embeddings.pkl`
- `process_incoming.py` – accepts a user query and generates a course-based answer
- `download_model.py` – downloads the Whisper model used for transcription
- `prompt.txt` – stores the latest prompt used for model inference

## Requirements

Before running the project, ensure the following are installed:

- Python 3.9+ or 3.10+
- FFmpeg available on your system PATH
- Ollama installed and running locally
- Internet access for downloading required models

### Python Dependencies

Install the required packages using:

```bash
pip install numpy pandas scikit-learn joblib requests faster-whisper huggingface_hub
```

### Ollama Models

Pull the required models:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

## Setup Instructions

1. Clone the repository:

```bash
git clone <https://github.com/rahulcoder96/RAG-based-AI-teaching-assistant>
cd RAG-based-AI-Teaching-Assistant
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
venv\\Scripts\\activate
```

3. Install the dependencies:

```bash
pip install numpy pandas scikit-learn joblib requests faster-whisper huggingface_hub
```

4. Start Ollama and pull the required models:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

5. Add your course videos to the `videos/` folder.

6. Run the pipeline:

```bash
python download_model.py
python process_video.py
python mp3_to_json.py
python preprocess_json.py
python process_incoming.py
```

## Example Questions

The assistant can answer questions such as:

- Where is the form tag explained?
- Which video explains CSS selectors?
- At what timestamp is the box model taught?

## Notes

- The assistant is intended for course-specific content and responds only to questions related to the available lesson material.
- Output quality depends on transcription accuracy, embedding quality, and the chosen LLM model.
- GPU support is recommended for faster transcription with larger audio files.
