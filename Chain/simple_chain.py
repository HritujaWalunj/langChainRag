from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
# import os, requests
from langchain_openai import ChatOpenAI

load_dotenv()

# resp = requests.get("https://router.huggingface.co/v1/models", 
#     headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"})

# for m in resp.json()["data"]:
#     print(m["id"])


# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",    
#     task="text-generation")


prompt = PromptTemplate(
    template="Generate 5 interesting facts about the {topic}",
    input_variables=["topic"]
)

# model = ChatHuggingFace(llm=llm)
model = ChatOpenAI(model="gpt-4.1-nano")

parser = StrOutputParser()
chain = prompt | model | parser

result = chain.invoke({'topic':'moon'})
print(result)