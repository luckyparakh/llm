
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

examples = [
    {"input": "0°C", "output": "32°F"},
    {"input": "10°C", "output": "50°F"},
    {"input": "20°C", "output": "68°F"},
    {"input": "30°C", "output": "86°F"},
    {"input": "40°C", "output": "104°F"},
]
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="User: {input}\nAI: {output}"
)
db = Chroma()
embedding = OpenAIEmbeddings()
# k is Number of examples to retrieve
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, embedding, db, k=2)

fsp = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Convert the temperature from Celsius to Fahrenheit",
    suffix="Input: {temperature}\nOutput:",
    input_variables=["temperature"],
)

print(fsp.format(temperature="40°C"))
print(fsp.format(temperature="50°C"))

fsp.example_selector.add_example({"input": "50°C", "output": "122°F"})

print(fsp.format(temperature="60°C"))

chain = fsp | llm
print(chain.invoke({"temperature": "60°C"}).content.strip())
