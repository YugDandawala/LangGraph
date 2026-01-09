from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import psycopg
import requests
import asyncio
import threading
import os
import atexit

load_dotenv()

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    model="mistralai/devstral-2512:free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol"""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    return requests.get(url, timeout=30).json()

# ---------- MCP tools ----------
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python3",
            "args": ["/Users/nitish/Desktop/mcp-math-server/main.py"],
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
        },
    }
)

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        print("⚠️ MCP tools failed:", e)
        return []

mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def chat_node(state: ChatState):
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

pg_conn = None
checkpointer = None

async def _init_checkpointer():
    global pg_conn, checkpointer

    pg_conn = await psycopg.AsyncConnection.connect(
        "postgresql://postgres:Admin1234@localhost:5432/langgraph"
    )

    await pg_conn.set_autocommit(True)

    checkpointer = AsyncPostgresSaver(pg_conn)
    await checkpointer.setup()

    print("✅ Postgres connected")
    return checkpointer

checkpointer = run_async(_init_checkpointer())

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

async def _alist_threads():
    threads = set()
    async for checkpoint in checkpointer.alist(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(threads)

def retrieve_all_threads():
    return run_async(_alist_threads())

async def _close_pg():
    if pg_conn:
        await pg_conn.close()
        print("🛑 Postgres closed")

def _shutdown():
    asyncio.run_coroutine_threadsafe(_close_pg(), _ASYNC_LOOP)

atexit.register(_shutdown)