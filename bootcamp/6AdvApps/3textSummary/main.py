from json import load
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st


def get_response(draft_input: str, openai_api_key: str):
    llm = OpenAI(temperature=.7, openai_api_key=openai_api_key)
    
    # The load_summarize_chain function in Langchain is a utility that simplifies the process of creating a summarization chain. It allows you to load a pre-defined summarization chain that can be used to summarize text documents efficiently. This function abstracts away some of the complexities involved in setting up the chain, making it easier for developers to implement summarization in their applications.

    # Type
    # • Stuff: Combines all documents into a single prompt and passes it to the LLM.
    # • Map-Reduce: Processes documents in batches, summarizes each batch, and then combines the summaries.
    # • Refine: Iteratively refines the summary based on the previous outputs.
    
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    # Split the text into chunks
    text = CharacterTextSplitter()
    chunks = text.split_text(draft_input)

    # Create list of docs
    docs = []
    for c in chunks:
        #  The Document class is typically used in Langchain to represent a piece of text along with any associated metadata.
        docs.append(Document(page_content=c))
    return chain.run(docs)


st.set_page_config(page_title="Text Summarizer")
st.title("Text Summarizer")
r = ""
with st.form("form", clear_on_submit=True):
    openai_api_key = st.text_input(
        "Enter your OpenAI API Key", type="password")
    draft_input = st.text_area(
        "Enter the text you want to summarize", height=200)
    submit = st.form_submit_button("Summarize")

    if submit and openai_api_key.startswith("sk-"):
        r = get_response(draft_input, openai_api_key)

if len(r) > 0:
    st.write(r)
