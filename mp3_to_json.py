import os
import sys

# Add the paths where pip installs the nvidia DLLs
package_path = os.path.join(sys.prefix, "Lib", "site-packages")
os.add_dll_directory(os.path.join(package_path, "nvidia", "cublas", "bin"))
os.add_dll_directory(os.path.join(package_path, "nvidia", "cudnn", "bin"))

# Now import faster_whisper

from faster_whisper import WhisperModel
import json
os.makedirs("jsons", exist_ok=True)

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

audios = os.listdir("audios")

for audio in audios: 
    if "_" in audio:
        number = audio.split("_")[0]
        title = os.path.splitext(audio.split("_")[1])[0]
        print(f"Processing: {number} - {title}")
        
        if int(number.split(" -")[0]) >= 39:
            segments, info = model.transcribe(
                audio=f"audios/{audio}",
                language="hi",
                task="translate",
                word_timestamps=False
            )
        
            chunks = []
            full_text_list = []
            for segment in segments:
                chunks.append({
                    "number": number, 
                    "title": title, 
                    "start": segment.start, 
                    "end": segment.end, 
                    "text": segment.text
                })
                full_text_list.append(segment.text)
            
            full_text = "".join(full_text_list)
            chunks_with_metadata = {"chunks": chunks, "text": full_text}

            output_filename = f"jsons/{audio}.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)
                
            print(f"Saved to {output_filename}")