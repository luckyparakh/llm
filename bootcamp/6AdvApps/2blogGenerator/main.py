from pprint import pformat
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

template="""
As experienced startup and venture capital writer, 
    generate a 400-word blog post about {topic}
    
    Your response should be in this format:
    First, print the blog post.
    Then, sum the total number of words on it and print the result like this: This post has X words.
    """
p=PromptTemplate(input=["topic"],template=template)

st.set_page_config(page_title="Blog Generator", page_icon="📝")
st.title("Blog Generator")
openai_api_key=st.sidebar.text_input("Enter your OpenAI API Key", type="password")
topic=st.text_input("Enter the topic of your blog post", key="topic")
if openai_api_key:
    llm = OpenAI(temperature=.7, openai_api_key=openai_api_key)
else:
    st.write("Please enter your OpenAI API Key")
    st.stop()
if topic:
    p_format=p.format(topic=topic)
    blog=llm(p_format)
    st.write(blog)
