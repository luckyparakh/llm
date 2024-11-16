from fastapi import FastAPI
from langserve import add_routes
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate

_ = load_dotenv(find_dotenv())

model = ChatGroq(model="Gemma2-9b-It")
print(model)

# Messages
messages = [
    HumanMessage(content="Hello, how are you"),
    SystemMessage(content="Translate from English to Spanish")
]
parser = StrOutputParser()
r = model.invoke(messages)
print(parser.invoke(r))

# LCEL
c = model | parser
print(c.invoke(messages))

# Prompt template
p = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate from English to {language}"),
        ("user", "{text}")
    ]
)
print(p.invoke({"language": "hindi", "text": "Hello"}))
# messages=[SystemMessage(content='Translate from English to hindi', additional_kwargs={},
# response_metadata={}), HumanMessage(content='Hello', additional_kwargs={}, response_metadata={})]

c = p | model | parser
print(c.invoke({"language": "hindi", "text": "Hello"}))

app = FastAPI(name="Langchain server", version="1.0",
              description="App with Langserve and GROQ")
add_routes(
    app=app,
    runnable=c,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
