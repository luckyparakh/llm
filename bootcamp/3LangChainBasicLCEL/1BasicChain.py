from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

chatModel = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template(
    "tell me amazing fact about {country}")
chain = prompt | chatModel | StrOutputParser()
r = chain.invoke({"country": "India"})
print(r)
print("\n\n")
for chunk in chain.stream({"country": "UAE"}):
    print(chunk, end="", flush=True)

print("\n\n")
r = chain.batch([{"country": "USA"}, {"country": "UK"}])
print(r)

# async invoke
# chain.ainvoke({"country": "USA"})