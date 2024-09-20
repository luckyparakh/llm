from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain 
from langchain.prompts import HumanMessagePromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory,ConversationSummaryMemory

load_dotenv()
chat=ChatOpenAI(verbose=True)

# This tells the memory object to use “message” as the key for storing conversation messages.
# return_messages specifies that when retrieving data from memory, it should return the messages.
# memory=ConversationBufferMemory(
#     memory_key="message",
#     return_messages=True,
#     chat_memory=FileChatMessageHistory('memory.json')
#     )

# There are many memory type one of them is ConversationBufferMemory with which FileChatMessageHistory memory works fine.
# But you can't save long conversation in file, for that something better is needed & there comes in ConversationSummaryMemory which is like another chain
# which takes model

# sometime gives bad answers
memory=ConversationSummaryMemory.from_messages(
    llm=chat,
    memory_key="message",
    return_messages=True,
    chat_memory=FileChatMessageHistory('memory.json')
)
prompt=ChatPromptTemplate(
    input_variables=["content","message"],
    messages=[
        MessagesPlaceholder(variable_name="message"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

# This is an instance of the LLMChain class you created earlier in the script. 
# It represents a sequence of operations that use a language model to process input and generate output.
chain=LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    content=input(">> ")

    # When you call chain like a function, you’re actually invoking its __call__ method. 
    # This method is designed to take input, send it to the language model, and get a response.
    # This is a Python dictionary with a key "content" and a value content. The key matches the input variable you defined in your prompt template (input_variables=["content"]). 
    # The value is whatever the user types in response to the input(">> ") prompt.
    result=chain({"content":content})
    print(result["text"])