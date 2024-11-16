
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

_ = load_dotenv(find_dotenv())

page = "https://www.mongodb.com/docs/mongodb-shell/crud/update/"
# page = "https://lilianweng.github.io/posts/2023-06-23-agent/"
# Load doc
loader = WebBaseLoader(
    web_path=page)
docs = loader.load()
# print(docs)

# Load to Vector DB
ts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = ts.split_documents(docs)
# print(len(chunks))

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(chunks, embeddings)
r = db.as_retriever()
# print(r)

query = "The MongoDB shell provides the following methods to update"
# print(db.similarity_search(query)[0].page_content)


p = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context
    <context>
    {context}
    </context>
    """
)

llm = OpenAI()
document_chain = create_stuff_documents_chain(llm, p)

# print(document_chain)
# bound=RunnableBinding(bound=RunnableAssign(mapper={
#   context: RunnableLambda(format_docs)
# }), kwargs={}, config={'run_name': 'format_inputs'}, config_factories=[])
# | ChatPromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, template='\n    Answer the following question based only on the provided context\n    <context>\n    {context}\n    </context>\n    '), additional_kwargs={})])
# | OpenAI(client=<openai.resources.completions.Completions object at 0x7f04258a4eb0>, async_client=<openai.resources.completions.AsyncCompletions object at 0x7f04258a6fb0>, model_kwargs={}, openai_api_key=SecretStr('**********'), logit_bias={})
# | StrOutputParser() kwargs={} config={'run_name': 'stuff_documents_chain'} config_factories=[]


rag_chain = create_retrieval_chain(r, document_chain)
print(rag_chain)
# RunnableAssign(mapper={
#   context: RunnableBinding(bound=RunnableLambda(lambda x: x['input'])
#            | VectorStoreRetriever(tags=['FAISS', 'OpenAIEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x75c1f69a4e80>, search_kwargs={}), kwargs={}, config={'run_name': 'retrieve_documents'}, config_factories=[])
# })
# | RunnableAssign(mapper={
#     answer: RunnableBinding(bound=RunnableBinding(bound=RunnableAssign(mapper={
#               context: RunnableLambda(format_docs)
#             }), kwargs={}, config={'run_name': 'format_inputs'}, config_factories=[])
#             | ChatPromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], input_types={}, partial_variables={}, template='\n    Answer the following question based only on the provided context\n    <context>\n    {context}\n    </context>\n    '), additional_kwargs={})])
#             | OpenAI(client=<openai.resources.completions.Completions object at 0x75c1f69a5210>, async_client=<openai.resources.completions.AsyncCompletions object at 0x75c1f69a7310>, model_kwargs={}, openai_api_key=SecretStr('**********'), logit_bias={})
result = rag_chain.invoke({"input": query})
print(result['answer'])
