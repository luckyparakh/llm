from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
_ = load_dotenv(find_dotenv())

llm = ChatGroq(model="Gemma2-9b-It")

sessions = {}


def get_session(session: str) -> BaseChatMessageHistory:
    if session not in sessions:
        sessions[session] = ChatMessageHistory()
    return sessions[session]


with_message_history = RunnableWithMessageHistory(llm, get_session)

# key should be session_id else it will give key error
config = {"configurable": {"session_id": "s1"}}
r = with_message_history.invoke("My name is RP", config)
print(r.content)
print("------------")
# key should be session_id else it will give key error
config = {"configurable": {"session_id": "s1"}}
r = with_message_history.invoke("What is my name", config)
print(r.content)
print("------------")
# key should be session_id else it will give key error
config = {"configurable": {"session_id": "s2"}}
r = with_message_history.invoke("What is my name", config)
print(r.content)


p = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant who talk in {language}"),
        MessagesPlaceholder("messages")
    ],
)
c = p | llm
with_message_history1 = RunnableWithMessageHistory(
    c, get_session, input_messages_key="messages")
r = with_message_history1.invoke(
    {"messages": "My name is Gemma", "language": "Spanish"}, config)
print("----------------")
print(r.content)

r = with_message_history1.invoke(
    {"messages": "What was my name", "language": "hindi"}, config)
print("----------------")
print(r.content)

# Trim messages
trimmer = trim_messages(
    token_counter=llm,
    max_tokens=70,
    start_on="human",
    include_system=True,
    strategy="last",
    allow_partial=False
)
messages = [
    SystemMessage(content="I am helpful assistant"),
    HumanMessage(content="Hi I am RP and I like Icecream"),
    AIMessage(content="Hi RP, which flavour?"),
    HumanMessage(content="Chocolate and what about you"),
    AIMessage(content="I don't like icecream"),
    HumanMessage(content="What is 2+3?"),
    AIMessage(content="it is 5"),
]
print("-------Trimmer---------")
print(trimmer.invoke(messages))
t = RunnablePassthrough.assign(
    messages=itemgetter("messages") | trimmer) | p | llm
wfm = RunnableWithMessageHistory(t, get_session, input_messages_key="messages")
r = wfm.invoke({"messages": messages + [HumanMessage(content="What is my name")],
                "language": "EN"}, config)
print(r.content)

# Trim messages
trimmer1 = trim_messages(
    token_counter=llm,
    max_tokens=30,
    start_on="human",
    include_system=True,
    strategy="last",
    allow_partial=False
)
print("-------Trimmer-1--------")
print(trimmer1.invoke(messages))
rpt = RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer1)
t = rpt | p | llm
wfm = RunnableWithMessageHistory(t, get_session, input_messages_key="messages")
messages.append(HumanMessage(content="Which ice cream I like"),)
r = wfm.invoke({"messages": messages, "language": "EN"}, config)
print(r.content)
