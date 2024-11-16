from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
_ = load_dotenv(find_dotenv())

llm = ChatGroq(model="llama3-8b-8192")
# llm = ChatGroq(model="Gemma2-9b-It")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
loader = WebBaseLoader(
    web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)
docs = loader.load()
# print(docs)
ts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = ts.split_documents(docs)

db = FAISS.from_documents(chunks, embeddings)
r = db.as_retriever(k=1)
sp = """
You are very good assistant who can answer question very well with given context: {context}.
If you don't know the answer then say so and also give answer in max 2 sentences.
"""
p = ChatPromptTemplate.from_messages(
    [
        ("system", sp), ("human", "{input}")
    ]
)
stuff_chain = create_stuff_documents_chain(llm, p)
# print(stuff_chain)
r_chain = create_retrieval_chain(r, stuff_chain)
# print(r_chain)
ans = r_chain.invoke({"input": "What are agents"})
print(ans['answer'])


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, r, contextualize_q_prompt
)
print("------------")
print(history_aware_retriever)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)
print("------------")
print(rag_chain)


config = {"configurable": {"thread_id": "abc123"}}

result = rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config=config,
)
print(result["answer"])