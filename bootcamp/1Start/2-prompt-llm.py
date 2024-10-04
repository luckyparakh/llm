
import re
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llmModel = OpenAI()
chatModel = ChatOpenAI(model="gpt-4o-mini")

promptTemplate = PromptTemplate.from_template(
    "Tell me {type} fact about {entity}")

llmModelPrompt = promptTemplate.format(type="fun", entity="India")
response = llmModel.invoke(llmModelPrompt)
print("-----------------------------")
print(response)

chatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are {profession} of Noida"),
    ("human", "Hello Mr. {profession}, I have a question."),
    ("ai", "Sure!"),
    ("human", "{user_input}"),
])
message = chatPromptTemplate.format_messages(
    profession="expert tour operator",
    user_input="I am looking for a good veg restaurants in the center of Mussoorie town.")
response = chatModel.invoke(message)
print("-----------------------------")
print(response)

examples = [
    {"input": "hi", "output": "hola"},
    {"input": "bye!", "output": "¡adiós!"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [('human', '{input}'), ('ai', '{output}')]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are expert european language translator'),
        few_shot_prompt,
        ('human', '{input}'),
    ]
)
# message = final_prompt.format(input="Good Morning!")
# response = chatModel.invoke(message)
# print("-----------------------------")
# print(response)

chain = final_prompt | chatModel
response = chain.invoke({"input":"good night"})
print("-----------------------------")
print(response)