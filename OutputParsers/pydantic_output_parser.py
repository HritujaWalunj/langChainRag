
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 128,
        "do_sample": False,
        "return_full_text": False,
    },
)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    city: str = Field(description="City of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "Generate a fictional person from {place}.\n"
        "Return only a JSON object with name, age, and city.\n"
        "{format_instructions}"
    ),
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

prompt = template.invoke({"place": "New York"})

# Generate only the new text, not the prompt
result = llm.invoke(prompt.to_string())

# Parse the generated JSON
parsed_result = parser.parse(result)

print("Parsed result:", parsed_result)
