from itertools import chain
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import CharacterTextSplitter

_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

with open("test.txt", "w") as f:
    f.write(text)


text_loader = TextLoader("test.txt")
data = text_loader.load()
print(len(data))

embedding = OpenAIEmbeddings()
ts = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = ts.split_documents(data)
print(len(docs))

db = Chroma.from_documents(documents=docs, embedding=embedding)
r = db.as_retriever()
c = RetrievalQA.from_chain_type(
    # Stuffing is one way to supply information to the LLM. Using this technique, we "stuff" all the information into the LLM's prompt. However, this method is only effective with shorter documents, as most LLMs have a context length limit.
    # Additionally, a similarity search is conducted using the embeddings to identify matching documents to be used as context for the LLM.
    chain_type="stuff",
    retriever=r,
    llm=llm,
)
print(c.run("How Google plans to challenge OpenAI?"))

# In the Q&A example, we cut the text into equal parts, causing both useful and useless text to show up when a user asks a question.
# Including unrelated information in the LLM prompt is detrimental because:
# It can divert the LLM's focus from pertinent details.
# It occupies valuable space that could be utilized for more relevant information.

# A DocumentCompressor abstraction has been introduced to address this issue, allowing compress_documents on the retrieved documents.

# The ContextualCompressionRetriever is a wrapper around another retriever in LangChain. It takes a base retriever and a DocumentCompressor and automatically compresses the retrieved documents from the base retriever. This means that only the most relevant parts of the retrieved documents are returned, given a specific query.

compressor = LLMChainExtractor.from_llm(llm)
cr = ContextualCompressionRetriever(base_retriever=r, base_compressor=compressor)
crc=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=cr,
    chain_type="stuff"
)
print("----------------------------")
print(c.run("How Google plans to challenge OpenAI?"))