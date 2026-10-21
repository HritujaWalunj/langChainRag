from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel 

load_dotenv()


model1 = ChatOpenAI(model="gpt-4.1-nano")
model2 = ChatOpenAI()

prompt1 = PromptTemplate(
    template="Generate a short and simple notes from the following topic \n {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 short question answer from the following topic \n {topic}",
    input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes ->{notes} and  quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser 

chain = parallel_chain | merge_chain

topic = """ A Large Language Model (LLM) is an artificial intelligence model designed to understand, process, and generate human language. It is trained on large amounts of text data, such as books, articles, websites, and code, to learn patterns and relationships between words.

LLMs commonly use a deep learning architecture called the Transformer, which uses a mechanism called attention to understand relationships between different parts of the input text. During text generation, an LLM typically predicts the next token based on the previous tokens and the patterns learned during training.

A token can be a word, part of a word, or a character. For example, ChatGPT, Llama, and Gemini are examples of LLM-based AI systems. LLMs can perform many tasks, including answering questions, summarizing documents, translating languages, writing emails, generating code, and explaining technical concepts.

An important concept in LLMs is the context window, which determines how many tokens the model can process as context in a single request. LLM responses are influenced by the prompt, model training, and generation settings.

LLMs can sometimes produce incorrect or misleading information, even when their answers sound confident. Therefore, their outputs should be verified when accuracy is important.

LLMs are widely used in chatbots, virtual assistants, document analysis, coding assistants, and Retrieval-Augmented Generation (RAG) applications, where models use relevant external information to answer questions."""

result = chain.invoke({'topic': topic})

print(result)

