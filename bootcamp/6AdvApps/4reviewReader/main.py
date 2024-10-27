import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

template = """
Read the given review {review} and answer the following questions:
1. What is the overall sentiment of the review? Give output as either positive, neutral or negative. If you can't find sentiment, write "Sentiment not found".
2. What is price perception of the reviewer? Give output as either expensive or cheap. If you can't find sentiment, write "Perception not found".

Your response should be in bullet point and in this format:
 - Sentiment
 - Price perception

Input example:
This dress is pretty amazing. It arrived in two days, just in time for my wife's anniversary present. It is cheaper than the other dresses out there, but I think it is worth it for the extra features.

Output example:
- Sentiment: Positive
- How long took it to deliver? 2 days
- How was the price perceived? Cheap

"""
p = PromptTemplate(input=["review"], template=template)

st.set_page_config(page_title="Review Reader", page_icon="📝")
st.title("Review Reader")
openai_api_key = st.text_input(
    "Enter your OpenAI API Key", type="password", placeholder="sk-...")
user_input = st.text_area("Enter the review you want to analyze", height=200)
if openai_api_key == "" or openai_api_key.startswith("sk-") == False:
    st.write("Please enter your OpenAI API Key")
    st.stop()
if user_input == "":
    st.write("Please enter the review you want to analyze")
    st.stop()
if len(user_input) < 10 or len(user_input) > 5000:
    st.write("Please enter a review with at least 100 characters and at most 500 characters")
    st.stop()
llm = OpenAI(temperature=.7, openai_api_key=openai_api_key)
p_format = p.format(review=user_input)
r = llm(p_format)
st.write(r)
