from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template(
    "tell me amazing fact about {country}")
chain = prompt | model | StrOutputParser()
print(chain.invoke({"country": "India"}))
print("-----------------------------")
# The bind() function in Langchain is used to set default or constant arguments for a Runnable object before it is invoked within a RunnableSequence.
chain = prompt | model.bind(stop=["India"]) | StrOutputParser()
print(chain.invoke({"country": "India"})) 


def make_uppercase(arg):
    return arg["original_input"].upper()

# The assign() function in Langchain is used to create a new Runnable object that assigns a name to the output of another Runnable object.
# The main purpose of assign() is to make it easier to access and reference the output of a specific step within a larger RunnableSequence.
chain = RunnableParallel({"original_input": RunnablePassthrough()}).assign(
    uppercase=RunnableLambda(make_uppercase))

response = chain.invoke("whatever")
print(response) # {'original_input': 'whatever', 'uppercase': 'WHATEVER'}
