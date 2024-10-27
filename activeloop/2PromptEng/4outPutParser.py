
from typing import List
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import PydanticOutputParser

_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """
Offer a list of suggestions to substitute the specified target_word based the presented context.
target_word={target_word}
context={context}
"""

p = PromptTemplate(
    input_variables=["target_word", "context"],
    template=template
)
c = p | llm
print(c.invoke({"target_word": "good",
      "context": "The movie was very good."}).content.strip())

'''
Here are some suggestions to substitute the word "good" in the context of the sentence "The movie was very good":

1. Excellent
2. Outstanding
3. Impressive
4. Enjoyable
5. Fantastic
6. Superb
7. Remarkable
8. Thrilling
9. Engaging
10. Entertaining

You can choose any of these alternatives based on the specific nuance you want to convey!
'''

# Above output is not necessarily comes in above format. It can be in any format.
parser = CommaSeparatedListOutputParser()

template_cs = """
Offer a list of suggestions to substitute the specified target_word based the presented context.
target_word={target_word}
context={context}
{format_instructions}
"""

p_cs = PromptTemplate(
    input_variables=["target_word", "context"],
    template=template_cs,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
cs = p_cs | llm
print("-------------------------------")
print(p_cs.format(target_word="good", context="The movie was very good."))
'''
Offer a list of suggestions to substitute the specified target_word based the presented context.
target_word=good
context=The movie was very good.
Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`
'''
r = cs.invoke(
    {"target_word": "good", "context": "The movie was very good."}).content.strip()
print("-------------------------------")
print(parser.parse(r))


class Suggestion(BaseModel):
    words: List[str] = Field(
        description="A list of suggestions to substitute the target word in the context.")
    
    @field_validator("words")
    def not_start_with_number(cls, v):
        # v: The value being validated, which is expected to be a list (or similar iterable) of words.
        for word in v:
            # Check 1st letter of word & it should not be digit
            if word[0].isdigit():
                raise ValueError("Suggestions should not start with a number.")
        return v
py_parser= PydanticOutputParser(pydantic_object=Suggestion) 

p_py="""
Offer a list of suggestions to substitute the specified target_word based the presented context.
{format_instructions}
target_word={target_word}
context={context}
""" 
p_py=PromptTemplate(input_variables=["target_word", "context"], template=p_py, partial_variables={"format_instructions": py_parser.get_format_instructions()})

c_py=p_py|llm
print("-------------------------------")
print(p_py.format(target_word="good", context="The movie was very good."))
print("-------------------------------")
r=c_py.invoke({"target_word": "good", "context": "The movie was very good."})
print(r)
print(py_parser.parse(r.content.strip()))
print(py_parser.parse(r.content.strip()).words)

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitute words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

    # Throw error in case of recieving a numbered-list from API
    @field_validator('words')
    def not_start_with_number(cls, field):
      for item in field:
        if item[0].isnumeric():
          raise ValueError("The word can not start with numbers!")
      return field
  
    # The end_with_dot method ensures that every string in the reasons field ends with a dot.
    @field_validator('reasons')
    def end_with_dot(cls, field):
      for idx, item in enumerate( field ):
        if item[-1] != ".":
          field[idx] += "."
      return field

py_parser_r= PydanticOutputParser(pydantic_object=Suggestions) 

p_py_r="""
Offer a list of suggestions to substitute the specified target_word based the presented context and gives reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
""" 
p_py_r=PromptTemplate(input_variables=["target_word", "context"], template=p_py_r, partial_variables={"format_instructions": py_parser_r.get_format_instructions()})
ch_r=p_py_r|llm
print("-------------------------------")
print(p_py_r.format(target_word="good", context="The movie was very good."))
print("-------------------------------")
r=ch_r.invoke({"target_word": "good", "context": "The movie was very good."})
print(r)
print(py_parser_r.parse(r.content.strip()))