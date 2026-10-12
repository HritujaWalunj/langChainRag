from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="Qwen/Qwen2.5-1.5B-Instruct")
text = "Delhi is the capital of India."
vector = embedding.embed_query(text)
print(str(vector))