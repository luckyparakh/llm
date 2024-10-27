import re
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
urls = ['https://beebom.com/what-is-nft-explained/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-download-gif-twitter/',
        'https://beebom.com/how-use-chatgpt-linux-terminal/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-save-instagram-story-with-music/',
        'https://beebom.com/how-install-pip-windows/',
        'https://beebom.com/how-check-disk-usage-linux/']

# use the selenium scraper to load the documents
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()

ts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = ts.split_documents(docs_not_splitted)
print(len(docs))
# print(docs[0])

embedding = OpenAIEmbeddings()
db = Chroma.from_documents(documents=docs, embedding=embedding)
retriever = db.as_retriever()
c = RetrievalQA.from_chain_type(
    retriever=retriever,
    llm=llm,
    chain_type="stuff"
)

print(c.run("How to check disk usage in linux?"))
