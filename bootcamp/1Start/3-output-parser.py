import re
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llmModel = OpenAI()
chatModel = ChatOpenAI(model="gpt-4o-mini")

json_prompt = PromptTemplate.from_template(
    "Return a json object with key as answer, which has answer to {question}")
json_parser = SimpleJsonOutputParser()
chain = json_prompt | llmModel | json_parser
response = chain.invoke({"question": "What is the capital of India?"})
print(response)


class MyOutput(BaseModel):
    answer: str = Field(description="Answer to the question")


parser = JsonOutputParser(pydantic_object=MyOutput)
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()},
)

chain = prompt | chatModel | parser
print(chain.invoke({"query": "What is the capital of India?"}))
