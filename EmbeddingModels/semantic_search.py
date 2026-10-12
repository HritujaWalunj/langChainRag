from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="Qwen/Qwen2.5-1.5B-Instruct")

documents =[ "Delhi is the capital of India.",
            "Mumbai is the financial capital of India.",
            "Kolkata is the cultural capital of India.",
            "Chennai is the capital of Tamil Nadu."
]

query = "capital of Tamil Nadu"


query_embedding = embedding.embed_query(query)
docs_embeddings = embedding.embed_documents(documents)

scores = cosine_similarities = cosine_similarity([query_embedding], docs_embeddings)[0]
print(scores)

index, score  = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(documents[index], score)

