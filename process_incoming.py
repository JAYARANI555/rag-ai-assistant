
# import pandas as pd 
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np 
# import joblib 
# import requests
# from xmlrpc import client
# from openai import OpenAI
# from config import api_key

# client = OpenAI(api_key=api_key)


# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"] 
#     return embedding

# def inference(prompt):
#     r = requests.post("http://localhost:11434/api/generate", json={
#         # "model": "deepseek-r1",
#         "model": "llama3.2",
#         "prompt": prompt,
#         "stream": False
#     })

#     response = r.json()
#     print(response)
#     return response

# def inference_openai(prompt):
#     response = client.responses.create(   
#         model="gpt-4o-mini",
#         input= prompt
#     )
#     return response.output_text

# #  def inference_openai(prompt):
     
# df = joblib.load('embeddings.joblib')


# incoming_query = input("Ask a Question: ")
# question_embedding = create_embedding([incoming_query])[0] 

# # Find similarities of question_embedding with other embeddings
# # print(np.vstack(df['embedding'].values))
# # print(np.vstack(df['embedding']).shape)
# similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# # print(similarities)
# top_results = 5
# max_indx = similarities.argsort()[::-1][0:top_results]
# # print(max_indx)
# new_df = df.loc[max_indx] 
# # print(new_df[["title", "number", "text"]])

# prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

# {new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
# ---------------------------------
# "{incoming_query}"
# User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
# '''
# with open("prompt.txt", "w") as f:
#     f.write(prompt)

# # response = inference(prompt)["response"]
# # print(response)

# response = inference_openai(prompt)


# with open("response.txt", "w") as f:
#     f.write(response)
# # for index, item in new_df.iterrows():
# #     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])


# 
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. CREATE EMBEDDING USING OLLAMA
# =========================================================

def create_embedding(text_list):
    """
    Create embeddings using Ollama's bge-m3 model.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "bge-m3",
                "input": text_list
            },
            timeout=120
        )

        # Check if request was successful
        response.raise_for_status()

        data = response.json()

        return data["embeddings"]

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to Ollama.")
        print("Make sure Ollama is running.")
        return None

    except requests.exceptions.RequestException as e:
        print("\nERROR while creating embedding:")
        print(e)
        return None

    except KeyError:
        print("\nERROR: 'embeddings' key not found in Ollama response.")
        print(response.text)
        return None


# =========================================================
# 2. GENERATE ANSWER USING OLLAMA
# =========================================================

def inference(prompt):
    """
    Generate answer using Ollama's llama3.2 model.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        # Check if request was successful
        response.raise_for_status()

        data = response.json()

        return data["response"]

    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to Ollama.")
        print("Make sure Ollama is running.")
        return None

    except requests.exceptions.RequestException as e:
        print("\nERROR while generating answer:")
        print(e)
        return None

    except KeyError:
        print("\nERROR: 'response' key not found in Ollama response.")
        print(response.text)
        return None


# =========================================================
# 3. CONVERT SECONDS INTO MM:SS FORMAT
# =========================================================

def format_timestamp(seconds):
    """
    Convert seconds into MM:SS format.

    Example:
    10 seconds    -> 00:10
    65 seconds    -> 01:05
    125 seconds   -> 02:05
    383.12 seconds -> 06:23
    """

    seconds = float(seconds)

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    return f"{minutes:02d}:{remaining_seconds:02d}"


# =========================================================
# 4. LOAD EMBEDDINGS
# =========================================================

print("=" * 60)
print("LOADING COURSE EMBEDDINGS")
print("=" * 60)

try:
    df = joblib.load("embeddings.joblib")

except FileNotFoundError:
    print("\nERROR: embeddings.joblib file was not found.")
    print("Make sure embeddings.joblib is in the same folder.")
    exit()

except Exception as e:
    print("\nERROR while loading embeddings.joblib:")
    print(e)
    exit()


print("\nEmbeddings loaded successfully.")
print("Total video chunks:", len(df))


# =========================================================
# 5. ASK USER QUESTION
# =========================================================

incoming_query = input("\nAsk a Question: ")

print("\nYour Question:")
print(incoming_query)

print("\nSearching relevant video sections...")


# =========================================================
# 6. CREATE EMBEDDING FOR USER QUESTION
# =========================================================

question_embedding = create_embedding(
    [incoming_query]
)

if question_embedding is None:
    print("\nCould not create embedding for your question.")
    exit()


# Get the first embedding
question_embedding = question_embedding[0]


# =========================================================
# 7. CALCULATE COSINE SIMILARITY
# =========================================================

try:

    similarities = cosine_similarity(
        np.vstack(df["embedding"]),
        [question_embedding]
    ).flatten()

except Exception as e:
    print("\nERROR while calculating similarity:")
    print(e)
    exit()


# =========================================================
# 8. GET TOP 5 RELEVANT VIDEO CHUNKS
# =========================================================

top_results = 5

# Get indexes of the 5 most similar chunks
max_indx = similarities.argsort()[::-1][:top_results]

# Use iloc because max_indx contains row positions
new_df = df.iloc[max_indx].copy()


# =========================================================
# 9. CONVERT START AND END TIMES
# =========================================================

new_df["start_time"] = new_df["start"].apply(
    format_timestamp
)

new_df["end_time"] = new_df["end"].apply(
    format_timestamp
)


# =========================================================
# 10. CREATE VIDEO CHUNKS FOR LLM
# =========================================================

video_chunks = new_df[
    [
        "title",
        "number",
        "start_time",
        "end_time",
        "text"
    ]
].to_json(
    orient="records"
)


# =========================================================
# 11. CREATE PROMPT
# =========================================================

prompt = f"""
You are an AI learning assistant for a Web Development course.

The student asked this question:

"{incoming_query}"

Below are the most relevant sections from the course videos:

{video_chunks}


IMPORTANT:

The timestamps provided above are already converted into MM:SS format.

For example:
"06:23" means 6 minutes and 23 seconds.

DO NOT convert the timestamps again.

DO NOT show raw timestamps in seconds.

Use the provided "start_time" and "end_time" values exactly as they are.


YOUR TASK:

Answer the student's question using ONLY the information provided
in the relevant video sections above.


FOLLOW THESE RULES:

1. Tell the student which video contains the answer.

2. Mention the exact video title.

3. Mention the video number.

4. Clearly show the relevant timestamp in this format:

   Start watching at approximately 06:23–06:25.

5. Explain briefly what is being taught at that timestamp.

6. If multiple timestamps are relevant, list each one separately.

7. If multiple videos are relevant, mention all relevant videos.

8. Always make the timestamp easy for a student to understand.

9. Guide the student to the exact part of the video they should watch.

10. If the question is unrelated to the Web Development course,
    say:

    "I can only answer questions related to this course."

11. Do not mention:
    - embeddings
    - cosine similarity
    - vector database
    - retrieval
    - subtitle chunks
    - internal technical processes

12. Do not make up video numbers, titles, timestamps, or explanations
    that are not present in the provided information.

13. Keep the answer clear, concise, and student-friendly.


Give the final answer now.
"""


# =========================================================
# 12. SAVE PROMPT
# =========================================================

with open(
    "prompt.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(prompt)


# =========================================================
# 13. GENERATE ANSWER
# =========================================================

print("\nGenerating answer...")

response = inference(prompt)


# Check response
if response is None:
    print("\nCould not generate an answer.")
    exit()


# =========================================================
# 14. SAVE RESPONSE
# =========================================================

with open(
    "response.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(response)


# =========================================================
# 15. DISPLAY FINAL ANSWER
# =========================================================

print("\n")
print("=" * 60)
print("ANSWER")
print("=" * 60)

print(response)

print("=" * 60)

print("\nThe answer has also been saved to:")
print("response.txt")