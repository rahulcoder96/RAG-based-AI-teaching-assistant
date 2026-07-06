import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    # Set a reasonable batch size (e.g., 32 or 64 strings per request)
    BATCH_SIZE = 32
    all_embeddings = []
    
    # Process the large list in smaller, bite-sized slices
    for i in range(0, len(text_list), BATCH_SIZE):
        batch = text_list[i : i + BATCH_SIZE]
        
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": batch
        })
        
        response_data = r.json()
        
        # Error handling check
        if "error" in response_data:
            print(f"Ollama Error at batch {i}: {response_data['error']}")
            continue
            
        batch_embeddings = response_data.get("embeddings", [])
        all_embeddings.extend(batch_embeddings)
        
    return all_embeddings

jsons=os.listdir("jsons")  #list all the jsons
print(jsons)
my_dict=[]
chunk_id=0


for json_file in jsons:
    with open(f"jsons/{json_file}", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Processing {json_file}...")
    
    # Extract the texts
    texts_to_embed = [chunk["text"] for chunk in data["chunks"]]

    # Create embeddings for the texts
    embeddings = create_embedding(texts_to_embed)

    for i,chunk in enumerate(data["chunks"]):
        chunk["chunk_id"]=chunk_id
        chunk["embedding"]=embeddings[i]
        chunk_id+=1
        my_dict.append(chunk)  


df=pd.DataFrame.from_records(my_dict)
#save this dataframe
joblib.dump(df, "embeddings.pkl")

