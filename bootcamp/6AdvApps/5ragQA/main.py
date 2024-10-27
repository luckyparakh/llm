import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain


def get_response(openai_api_key, uploaded_file, query_text, response_text):
    llm = OpenAI(temperature=.7, openai_api_key=openai_api_key)
    text = uploaded_file.read().decode()
    text_splitter = CharacterTextSplitter( chunk_size=1000,
        chunk_overlap=0)
    chunks = text_splitter.create_documents(text)
    e = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(chunks, e)
    r = db.as_retriever()
    # create a real QA dictionary
    real_qa = [
        {
            "question": query_text,
            "answer": response_text
        }
    ]
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=r, chain_type="stuff", input_key="question")
    prediction = retrieval_chain.apply(real_qa)
    print(prediction)
    qa_chain = QAEvalChain.from_llm(llm=llm)
    graded_outputs = qa_chain.evaluate(examples=real_qa, predictions=prediction,
                                       question_key="question", answer_key="answer",
                                       prediction_key="result")
    print(graded_outputs)
    response = {
        "predictions": prediction,
        "graded_outputs": graded_outputs
    }

    return response


st.set_page_config(
    page_title="Evaluate a RAG App"
)
st.title("Evaluate a RAG App")

with st.expander("Evaluate the quality of a RAG APP"):
    st.write("""
        To evaluate the quality of a RAG app, we will
        ask it questions for which we already know the
        real answers.
        
        That way we can see if the app is producing
        the right answers or if it is hallucinating.
    """)

uploaded_file = st.file_uploader(
    "Upload a .txt document",
    type=["pdf", "txt"]
)

query_text = st.text_input(
    "Enter a question you have already fact-checked:",
    placeholder="Write your question here",
    disabled=not uploaded_file
)

response_text = st.text_input(
    "Enter the real answer to the question:",
    placeholder="Write the confirmed answer here",
    disabled=not uploaded_file
)

with st.form("form", clear_on_submit=True):
    openai_api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        disabled=not (uploaded_file and query_text and response_text)
    )
    submit = st.form_submit_button("Evaluate", disabled=not (
        uploaded_file and query_text and response_text))
    if submit and openai_api_key.startswith("sk-"):
        with st.spinner("Evaluating..."):
            response = get_response(
                openai_api_key, uploaded_file, query_text, response_text)
            st.write(response)
