from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts.example_selector import LengthBasedExampleSelector

_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

examples = [
    {
        "input": "How can I remember Capital of Australia?",
        "output": "Capital of Australia is Canberra. Kangaroo is a symbol of Australia. To remember it, think of the word 'Can' in Canberra and berra. So think of a can of beer served by a Kangaroo."
    },
    {
        "input": "How can I remember Capital of Russia?",
        "output": "Capital of Russia is Moscow. To remember it, think of the word 'Mos' in Moscow and cow. So think of a cow in Moscow on road and due to this people of Russia, 'rus gaya' (got angry)."
    }
]

example_template = """
User: {input},
AI: {output}
"""

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template
)
example_selector= LengthBasedExampleSelector(
    example_prompt=example_prompt,
    max_length=75,
    examples=examples
)

prefix = """The following are excerpts from conversations with an AI
assistant. 
 
The assistant is expert memory trainer. 
Assistant helps you remember things by providing you with interesting one liners.
One liner should include clues to remember the capital and country. 
Assistant is expert of both English and Hindi languages. And can provide you with memory tricks in both languages or mixing these.
Here are some examples:
"""

# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

fsp = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)
print(fsp.format(query="How can I remember Capital of India?"))
chain = fsp | llm
print(chain.invoke({"query": "How can I remember Capital of India?"}))
