from langchain_openai import ChatOpenAI
import os
from langchain_openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

llmModel = OpenAI()

response = llmModel.invoke("Tell me fun fact about India")
print(response)


chatModel = ChatOpenAI(model="gpt-4o-mini")
message = [
    ("system", "You are expert guide of Noida"),
    ("human", "I am looking for a good veg restaurant in the center of town."),
]

# To print whole message at once
response=chatModel.invoke(message)
print(response)

#  To stream
for chunk in chatModel.stream(message):
    print(chunk, end="", flush=True)
    print("\n")