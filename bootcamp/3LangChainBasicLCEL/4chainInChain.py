from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
prompt = ChatPromptTemplate.from_template(
    "tell me a sentence about {politician}")

model = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | model | StrOutputParser()

response = chain.invoke("Modi")
print(response)

historian_prompt = ChatPromptTemplate.from_template(
    "Is {politician} positive for Humanity? Give answer in yes or no")
chain2 = {"politician": chain} | historian_prompt | model | StrOutputParser()

response = chain2.invoke({"politician":"Modi"})
print("----------------------------")
print(response)
