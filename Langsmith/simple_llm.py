from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatOllama(model="llama3.2:3b", temperature=0.1)

parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)