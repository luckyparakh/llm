from langchain.chains import SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
_=load_dotenv(find_dotenv())
llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)
fp=PromptTemplate(
    input_variables=["country"],
    template="What is the capital of {country}"
)
sp=PromptTemplate(
    input_variables=["capital"],
    template="Provide brief about {capital}"
)

# fc=fp|llm
# print(fc.invoke({"country":"India"}))
# sc=sp|llm
# # chain=SimpleSequentialChain(chains=[fc,sc],verbose=True)
# chain=fc|sc
# # r=chain.run("India")
# r=chain.invoke({"country":"India"})
# print(r)

