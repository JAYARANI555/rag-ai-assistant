import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    
    response_json = r.json()
    
    # Error checking to see exactly what Ollama is complaining about if it fails
    if "embeddings" not in response_json:
        print(f"Ollama Error Response: {response_json}")
        raise KeyError(f"Ollama failed to generate embeddings. Response received: {response_json}")
        
    return response_json["embeddings"]


jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    
    # FIX: Loop through chunks individually so we don't overwhelm the Ollama API context limit
    for chunk in content['chunks']:
        try:
            # Pass a single chunk text as a list element
            embedding = create_embedding([chunk['text']])[0]
            
            chunk['chunk_id'] = chunk_id
            chunk['embedding'] = embedding
            chunk_id += 1
            my_dicts.append(chunk)
        except Exception as e:
            print(f"Skipping a problematic chunk in {json_file} due to error: {e}")
            continue

df = pd.DataFrame.from_records(my_dicts)
# Save this dataframe
joblib.dump(df, 'embeddings.joblib')
print("Successfully generated and saved all embeddings!")