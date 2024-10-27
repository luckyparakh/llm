from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
template = """
Question: {question}
Answer:
"""
p = PromptTemplate(
    input_variables=["question"],
    template=template
)

hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-large',
    model_kwargs={'temperature': 0}
)

chain = p | hub_llm
print(chain.invoke({"question": "What is the capital of France?"}))

template = """
Answer one question at a time.Questions are separated by '\n'.
Question: {question}
Answer:
"""

pm = PromptTemplate(
    input_variables=["question"],
    template=template
)
cm = pm | hub_llm
qs_str = (
    "What is the capital city of India?\n" +
    "What is the largest mammal on Earth?\n" +
    "Which gas is most abundant in Earth's atmosphere?\n" +
    "What color is a ripe banana?\n"
)
print(cm.invoke({"question": qs_str}))

pt = PromptTemplate(input_variables=["source", "target","text"], 
                    template="Translate {text} from {source} to {target} language.")

chain = pt | hub_llm
print(chain.invoke({"source": "English", "target": "Spanish","text":"Hello, how are you?"}))
