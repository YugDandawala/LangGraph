from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import requests
import os

os.environ["LANGCHAIN_PROJECT"] = "Langchain Agent"
load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=key&query={city}'

  response = requests.get(url)

  return response.json()

llm = ChatOllama(model="llama3.2:3b", temperature=0.1)

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  

agent = create_react_agent(
    llm = llm,
    tools = [search_tool, get_weather_data],
    prompt = prompt,
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = [search_tool, get_weather_data],
    verbose = True,
    max_iterations = 3,
    early_stopping_method = "generate"
)

response = agent_executor.invoke({"input": "What is the temperature of surat"})
print(response)
# Who is the director of movie Avatar?
# What is the current temp of surat
# Identify the birthplace city of director of movie interstellar(search) and give its current temperature.

print(response['output'])