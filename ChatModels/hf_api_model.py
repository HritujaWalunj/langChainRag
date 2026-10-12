from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os, requests

load_dotenv()

resp = requests.get("https://router.huggingface.co/v1/models", 
    headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"})

for m in resp.json()["data"]:
    print(m["id"])


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",    
    task="text-generation")

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is Capital of India?")
print(result)