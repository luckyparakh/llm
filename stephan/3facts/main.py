from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
embeddings=OpenAIEmbeddings()
splitter=CharacterTextSplitter(
    chunk_overlap=0,
    chunk_size=200,
    separator="\n"
)
loader=TextLoader("facts.txt")
docs=loader.load_and_split(
    text_splitter=splitter
)
# for doc in docs:
#     print(doc.page_content)
#     print("\n")

# Below code will create embeddings by calling OpenAI
db=Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="emb"
)

results=db.similarity_search(
    query="What is interesting fact about English language?",
    k=3
)
print(results)
for result in results:
    print("\n")
    print(result.page_content)
