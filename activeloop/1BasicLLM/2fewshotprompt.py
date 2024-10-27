from sys import prefix
from tabnanny import verbose
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
llm = OpenAI()
examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    },
]

# The example_prompt is a template that defines how each example input-output pair should be formatted in the final prompt that will be sent to the language model (LLM).
example_template = """
User: {query}
AI: {answer}
"""
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)
# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

fsp = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

# Prompt will look like
# The following are excerpts from conversations with an AI
# assistant. The assistant is known for its humor and wit, providing
# entertaining and amusing responses to users' questions. Here are some
# examples:

# User: What's the weather like?
# AI: It's raining cats and dogs, better bring an umbrella!
# User: How old are you?
# AI: Age is just a number, but I'm timeless.

# User: What's the weather like?
# AI: 

chain = LLMChain(llm=llm, prompt=fsp,verbose=True)
print(chain.invoke({"query": "What's the weather like in?"}))
