from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="Qwen/Qwen2.5-1.5B-Instruct")

documents =[ "Delhi is the capital of India.",
            " Mumbai is the financial capital of India.",
            " Kolkata is the cultural capital of India."
]


vector = embedding.embed_documents(documents)
print(str(vector))