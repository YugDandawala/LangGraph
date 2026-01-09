from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM'
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatOllama(model="llama3.2:3b", temperature=0.1)
model2 = ChatOllama(model="llama3.2:3b", temperature=0.2)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'run_name':"Sequential App",
    'tags':['Sequential app','report-generation','summarization'],
    'metadata':{'model1':"llama3.2","model2":"tempe0.2"}  
}
result = chain.invoke({'topic': 'Importance of sports in life'},config = config)

print(result)