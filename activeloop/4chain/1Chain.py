from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
_=load_dotenv(find_dotenv())
llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)
template="""
Offer list of max ten suggestions to substitute the specified target_word: {word}.
{output}
"""

prompt = PromptTemplate(
    input_variables=["word"],
    template=template,
    partial_variables={"output": CommaSeparatedListOutputParser().get_format_instructions}
)

c=prompt | llm
r=c.invoke({"word":"good"})
print(r)