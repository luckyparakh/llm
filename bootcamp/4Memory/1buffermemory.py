from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

_ = load_dotenv(find_dotenv())


llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."),
        HumanMessagePromptTemplate.from_template("{question}"),
        # Variable Name is chat_history and it should align with the memory variable name
        MessagesPlaceholder(variable_name="chat_history")
    ]
)
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)
