from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

_ = load_dotenv(find_dotenv())
p = ChatPromptTemplate.from_messages([
    ("system", "Translate following into {lang}:"),
    ("human", "{input}")
])
llm = ChatOpenAI(model="gpt-4o-mini")
chain = p | llm | StrOutputParser()
chain.invoke({
    "lang": "es",
    "input": "Hello, how are you?"
})

app = FastAPI(
    title="simpleTranslator",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)
# http://localhost:8000/translate/playground/
add_routes(app, chain, path="/translate")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
