import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests

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


def inference(prompt):
    r=requests.post("http://localhost:11434/api/generate", json={
        "model":"llama3.2"
        ,"prompt":prompt,
        "stream":False})
    
    response=r.json()
    return response


df=joblib.load("embeddings.pkl")

incoming_query=input("Enter your query: ")
question_embedding=create_embedding([incoming_query])[0]
# print(question_embedding)

# print(np.vstack(df["embedding"].values))
# print(np.vstack(df["embedding"].shape))

similarties=cosine_similarity([question_embedding], np.vstack(df["embedding"])).flatten()

print(similarties)
top_results=5
max_indx=similarties.argsort()[::-1][0:top_results]  # Get the indices of the top 3 most similar chunks    

print(max_indx)
new_df=df.iloc[max_indx]
# print(new_df[["number","title","text"]])

prompt=f'''I am teaching web development using sigma web development course. here are video
subtitles chunks containing video title,video number,start time in seconds, end time in
seconds, the text at that time:

{new_df[["number","title","start","end","text"]].to_json(orient="records")}
---------------------------------------------
"{incoming_query}"
user asked this question related to the video chunks,you have answer where and how much
 content is taught where(in which video and at what timestamp) and guide the user to go 
 that particular video.if user asks unrelated question,tell him that you can only answer
 questions related to the course.
'''

with open("prompt.txt","w",encoding="utf-8") as f:
    f.write(prompt)

result=inference(prompt)
print(result["response"])

# for index,item in new_df.iterrows():
#     print(index,item["title"],item["number"],item["text"],item["start"],item["end"])
