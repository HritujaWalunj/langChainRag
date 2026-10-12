from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={"temperature": 1, "max_length": 100})

model = ChatHuggingFace(llm=llm)

result = model.invoke("write a poem about the moon")
print(result.content)