from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat=ChatOpenAI(model="gpt-4o-mini")

ChatPromptTemplate(messages=[
    SystemMessagePromptTemplate("You are a nice chatbot that is expert at summarization."),
    HumanMessagePromptTemplate.from_template(),
    MessagesPlaceholder(variable_name="chat_history")
])