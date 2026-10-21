from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel , RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from langchain_protocol import Literal
from pydantic import BaseModel, Field


load_dotenv()

model1 = ChatOpenAI()

parser1= StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the Sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

Prompt1 = PromptTemplate(
    template="Classify the following feedback into positive or negative sentiment: \n {feedback} \n {format_instructions}",
    partial_variables={"format_instructions": parser2.get_format_instructions()},
    input_variables=["feedback"]
)

classifier_chain = Prompt1 | model1 | parser2

Prompt2 = PromptTemplate(
    template = "write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"]
)

Prompt3 = PromptTemplate(
    template = "write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', Prompt2 | model1 | parser1),
    (lambda x: x.sentiment == 'negative', Prompt3 | model1 | parser1),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "I love the new features in your product! It's made my life so much easier."})
# result = chain.invoke({"feedback": "I am disappointed with the recent update. It has caused several issues and made the app less user-friendly."})
print(result)