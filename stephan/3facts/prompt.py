from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

langchain.debug=True

load_dotenv()
chat=ChatOpenAI()
embeddings=OpenAIEmbeddings()

# Reading from DB
db=Chroma(
    embedding_function=embeddings,
    persist_directory="emb"
)
# retriever = db.as_retriever()
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)
chain=RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=chat,
    retriever=retriever
)
result=chain.run("What is interesting fact about English language?")
print(result)
