from huggingface_hub import snapshot_download
import os

print("Starting manual model download...")

# This forces the download of the exact model faster-whisper needs
model_dir = snapshot_download(
    repo_id="Systran/faster-whisper-large-v2",
    repo_type="model",
    force_download=False  # Resumes if it gets interrupted
)

print(f"\n Success! Model downloaded and cached locally at:\n{model_dir}")

