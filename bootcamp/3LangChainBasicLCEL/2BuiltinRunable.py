from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

chain = RunnablePassthrough()
print(chain.invoke("Hello World"))


def greet(name: str) -> str:
    return "Hello " + name


chain = RunnablePassthrough() | RunnableLambda(greet)
print(chain.invoke("Ram"))

chain = RunnableParallel({
    "operation_a": RunnablePassthrough(),
    "operation_b": RunnableLambda(greet)
})
print(chain.invoke("Ram"))

chain = RunnableParallel({
    "operation_a": RunnablePassthrough(),
    "operation_b": lambda x: "Hello " + x["name"]
})
print(chain.invoke({"name": "Ramesh", "name1": "Ram"}))

chatModel = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template(
    "tell me amazing fact about {country}")

chain = RunnableParallel({
    "operation_a": RunnablePassthrough(),
    "country": lambda x: "Sri" + x["name"]
}) | prompt | chatModel | StrOutputParser()
print(chain.invoke({"name": "Lanka", "name1": "Ram"}))


vectorstore = FAISS.from_texts(["My name is Elon Musk and I work at Fesla"],
                               embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_template(
    "Answer the question based on only following context: {context}. \                                           Question: {question}")
chain = RunnableParallel({"context": retriever, "question": RunnablePassthrough(
)}) | prompt | chatModel | StrOutputParser()
print(chain.invoke("What is your name? And where do you work?"))

prompt = ChatPromptTemplate.from_template("Answer the question based on only following context: {context}. \
                                            Question: {question}. Answer in following language: {language}")
# chainIG = RunnableParallel({"context": retriever, "question": itemgetter(
#     "question"), "language": itemgetter("language")}) | prompt | chatModel | StrOutputParser()
chainIG = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | chatModel
    | StrOutputParser()
)
r = chainIG.invoke(
    {"question": "What is my Name and where do I work?", "language": "Hindi"})
print(r)
