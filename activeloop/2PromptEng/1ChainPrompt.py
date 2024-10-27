from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_q = PromptTemplate(
    template="Who was the scientist who discovered the theory of relativity?", input_variables=[])

prompt_fact = PromptTemplate(
    template="Tell me more about {scientist} who discovered the theory of relativity.", input_variables=["scientist"])
chain_q = prompt_q | llm
r_q=chain_q.invoke({})
print(r_q.content.strip())
scientist=r_q.content.strip()
chain_f= prompt_fact | llm
r_f=chain_f.invoke({"scientist":scientist})
print("------------------------------------------")
print(r_f.content.strip())
