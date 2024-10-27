
from langchain.callbacks import get_openai_callback
from langchain_openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, RetrievalQA, LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.tools import DuckDuckGoSearchRun

_ = load_dotenv(find_dotenv())

llm = OpenAI()


def basic():
    text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
    print(llm.invoke(text))


def chain():
    p = PromptTemplate(
        input_variables=["product", "country"],
        template="Which company makes best {product} in {country}?"
    )
    c = p | llm | StrOutputParser()
    r = c.invoke({"product": "shoe", "country": "Bharat"})
    print(r)


def memory():
    c = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),
        verbose=True,
    )
    c.invoke("Hello")
    c.invoke("How are you?")
    print(c)  # Current Conversation section in output shows the conversation history


def vector_store():
    text = [
        "Napoleon Bonaparte was born in 15 August 1769",
        "Louis XIV was born in 5 September 1638"
    ]
    ts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    doc = ts.create_documents(text)
    embeddings = OpenAIEmbeddings()

    # Deep Lake
    # my_activeloop_org_id = "luckyparakh"
    # my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
    # dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    # db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    # db.add_documents(doc)
    db = FAISS.from_documents(doc, embeddings)

    rqa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
    )
    print(rqa.run("When was Napoleon Bonaparte born?"))
    # Now use the rqa chain to ask questions
    Tools = [Tool(
        name="RQA System",
        func=rqa.run,
        description="Ask a question and get an answer",
    ),]

    #  In this case, AgentType.ZERO_SHOT_REACT_DESCRIPTION indicates that the agent will use a zero-shot reasoning approach with a reactive description. This means the agent will generate responses based on the tools available without prior training on specific tasks.
    agent = initialize_agent(tools=Tools, llm=llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    print(agent.run("When was Napoleon Bonaparte born?"))
    print("---------------Will fail to find---------------------------------------")
    print(agent.run("When was Lady Gaga born?"))

    # Add more doc
    # create new documents
    texts = [
        "Lady Gaga was born in 28 March 1986",
        "Michael Jeffrey Jordan was born in 17 February 1963"
    ]
    new_docs = ts.create_documents(texts)
    db.add_documents(new_docs)
    print("-------------------------------------------------------------------")
    print(rqa.run("When was Lady Gaga born?"))
    print("-------------------------------------------------------------------")
    print(agent.run("When was Lady Gaga born?"))


def summary(query):
    prompt = PromptTemplate(
        input_variables=["abc"],
        template="Summarize the following text: {abc}"
    )
    summarizer = prompt | llm | StrOutputParser()
    return summarizer.invoke({"abc": query})


def search():
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Summarize the  following text: {query}"
    )
    summarizer = LLMChain(llm=llm, prompt=prompt)
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="Search on DuckDuckGo",
            func=search.run,
            description="Search for information on the web"
        ),
        Tool(
            name="Summarizer",
            # func=summarizer.run,
            func=summary,
            description="Summarizer the text",
        ),
    ]
    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, max_iterations=6)
    r = agent.run("Where is Aditya L1?")
    print(r)


def callback():
    with get_openai_callback() as cb:
        result = llm("Tell me a joke")
        print(cb)


# basic()
# chain()
# memory()
# vector_store()
# search()
callback()
