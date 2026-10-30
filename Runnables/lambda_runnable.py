from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a small joke on  {topic}",
    input_variables=['topic']
)

model = ChatOpenAI(model='gpt-4.1-mini')

parser = StrOutputParser()

def word_counter(str):
    return len(str.split())

RunnableLength = RunnableLambda(word_counter)

joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'count': RunnableLength
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

final_result = """ {} \n Word count - {}""".format(result['joke'], result['count'])

print(final_result)