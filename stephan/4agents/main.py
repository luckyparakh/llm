from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import OpenAIFunctionsAgent,AgentExecutor
from langchain.memory import ConversationBufferMemory
from tools.sql import run_query,get_tables,describe_tables
from tools.report import write_report_tool
from handlers.on_chat_start import OnChatStart

load_dotenv()
tables=get_tables()

memory=ConversationBufferMemory(return_messages=True,memory_key="chat_history")

chat=ChatOpenAI(
    callbacks=[OnChatStart()]
)
prompt=ChatPromptTemplate(
    # The list of messages in the ChatPromptTemplate is used to define the sequence and structure of the conversation prompts that the language model will follow. Each element in the list represents a different part of the conversation, allowing for a more organized and flexible way to manage the interaction flow.
    # The first message is a HumanMessagePromptTemplate that takes user input.
    # The second message is a MessagesPlaceholder for the agent's scratchpad, which can be used to store intermediate results or context.
    # This structure allows the ChatPromptTemplate to handle dynamic and context-aware conversations effectively.
    messages=[
        SystemMessage(content=(
            "Hello! You are a AI and have access to a SQLite Database.\n"
            f"Tables in Database are: {tables}\n"
            "Do not assume about what tables and columns are present in the database."
            "Instead use 'describe_tables' function."
            )
            ),
        MessagesPlaceholder(variable_name="chat_history"),
        # The first message is a human message prompt template with a placeholder for input.
        # The value for the {input} placeholder in the HumanMessagePromptTemplate is provided when the agent_executor is called with an argument
        # Input is also special variable, changing name of variable gives error
        HumanMessagePromptTemplate.from_template("{input}"),
        # A placeholder for the agent scratchpad (agent_scratchpad is special name)
        # The agent_scratchpad serves as a placeholder for storing intermediate messages or data that the agent might need to reference or use during its execution. 
        # It allows the agent to keep track of its state and any relevant information that accumulates as it processes the input and generates responses.
        # In the provided code, MessagesPlaceholder(variable_name="agent_scratchpad") is used to define this placeholder within the ChatPromptTemplate. This means that during the execution of the agent, any intermediate messages or data can be stored in and retrieved from agent_scratchpad.
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

tools=[run_query,describe_tables,write_report_tool]
# The agent is responsible for generating responses based on the provided prompt and tools. It uses the tools to perform specific tasks or queries that are part of its response generation process.
agent=OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

# The AgentExecutor is responsible for managing the execution of the agent. It ensures that the agent's responses are generated and executed correctly.
agent_executor=AgentExecutor(
    agent=agent,
    # verbose=True,
    tools=tools,
    memory=memory
)
# agent_executor("How many users have provided address?")
# agent_executor("How many users are there in database?")
# agent_executor("Summarize the top 5 most popular product.Write the results to a report file")
agent_executor("How many orders are there? Write the results to a report file")
agent_executor("Repeat the exact same for users")