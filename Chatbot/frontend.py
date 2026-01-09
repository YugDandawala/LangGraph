import streamlit as st
from backend import chatbot, retrieve_all_threads, ingest_pdf, thread_document_metadata
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
import uuid
import os

os.environ["LANGCHAIN_PROJECT"] = (
    "Chatbot(Streaming+ResumeChat+DatabaseStorage+Observability+tools+Rag)"
)
load_dotenv()

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    """Create a new chat thread explicitly"""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    """Load messages from LangGraph state"""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def get_thread_display_name(thread_id):
    """Derive chat name from first user message (persistent)"""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content[:30]

    return "Empty Chat"

# ------------ Session Setup -------------
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_id" not in st.session_state:
    if st.session_state["chat_threads"]:
        st.session_state["thread_id"] = st.session_state["chat_threads"][-1]
    else:
        new_id = generate_thread_id()
        st.session_state["thread_id"] = new_id
        st.session_state["chat_threads"].append(new_id)

if "message_history" not in st.session_state:
    messages = load_conversation(st.session_state["thread_id"])
    st.session_state["message_history"] = [
        {
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content,
        }
        for msg in messages
    ]

# ------- Sidebar UI --------
st.sidebar.title("LangGraph Chatbot")

# 🔥 PDF UPLOADER SECTION (ONLY NEW PART)
st.sidebar.divider()
st.sidebar.subheader("📄 Document Upload")

current_thread = st.session_state["thread_id"]
doc_meta = thread_document_metadata(current_thread)

if doc_meta:
    st.sidebar.success(f"Attached: {doc_meta.get('filename')}")
    st.sidebar.caption(
        f"Pages: {doc_meta.get('documents')} | Chunks: {doc_meta.get('chunks')}"
    )
else:
    st.sidebar.info("No document uploaded for this chat.")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    key=f"pdf_{current_thread}",
)

if uploaded_file is not None:
    with st.sidebar.status("📥 Ingesting PDF into vector database...", expanded=True):
        file_bytes = uploaded_file.read()
        meta = ingest_pdf(
            file_bytes=file_bytes,
            thread_id=current_thread,
            filename=uploaded_file.name,
        )
        st.write("✅ Ingestion complete")
        st.json(meta)

    st.rerun()

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for thread_id in reversed(st.session_state["chat_threads"]):
    display_name = get_thread_display_name(thread_id)

    if st.sidebar.button(display_name, key=f"thread_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        st.session_state["message_history"] = [
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content,
            }
            for msg in messages
        ]
        st.rerun()

# --------- Main ---------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "tool_bot",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
    st.rerun()

# import streamlit as st
# from backend import chatbot, retrieve_all_threads,ingest_pdf,thread_document_metadata
# from langchain_core.messages import HumanMessage, AIMessage ,ToolMessage
# from dotenv import load_dotenv
# import uuid
# import os

# os.environ["LANGCHAIN_PROJECT"] = "Chatbot(Streaming+ResumeChat+DatabaseStorage+Observability+tools+Rag)"
# load_dotenv()

# def generate_thread_id():
#     return str(uuid.uuid4())

# def reset_chat():
#     """Create a new chat thread explicitly"""
#     thread_id = generate_thread_id()
#     st.session_state["thread_id"] = thread_id
#     st.session_state["message_history"] = []
#     st.session_state["chat_threads"].append(thread_id)

# def load_conversation(thread_id):
#     """Load messages from LangGraph state"""
#     state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
#     return state.values.get("messages", [])

# def get_thread_display_name(thread_id):
#     """Derive chat name from first user message (persistent)"""
#     state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
#     messages = state.values.get("messages", [])

#     for msg in messages:
#         if isinstance(msg, HumanMessage):
#             return msg.content[:30]

#     return "Empty Chat"

# # ------------ Session Setup -------------
# if "chat_threads" not in st.session_state:
#     st.session_state["chat_threads"] = retrieve_all_threads()

# if "thread_id" not in st.session_state:
#     if st.session_state["chat_threads"]:
#         # Reopen last chat instead of creating a new one
#         st.session_state["thread_id"] = st.session_state["chat_threads"][-1]
#     else:
#         # First ever chat
#         new_id = generate_thread_id()
#         st.session_state["thread_id"] = new_id
#         st.session_state["chat_threads"].append(new_id)

# if "message_history" not in st.session_state:
#     messages = load_conversation(st.session_state["thread_id"])
#     st.session_state["message_history"] = [
#         {
#             "role": "user" if isinstance(msg, HumanMessage) else "assistant",
#             "content": msg.content,
#         }
#         for msg in messages
#     ]
# # ------- Sidebar UI --------
# st.sidebar.title("LangGraph Chatbot")

# if st.sidebar.button("New Chat"):
#     reset_chat()
#     st.rerun()

# st.sidebar.header("My Conversations")

# for thread_id in reversed(st.session_state["chat_threads"]):
#     display_name = get_thread_display_name(thread_id)

#     if st.sidebar.button(display_name, key=f"thread_{thread_id}"):
#         st.session_state["thread_id"] = thread_id
#         messages = load_conversation(thread_id)
#         st.session_state["message_history"] = [
#             {
#                 "role": "user" if isinstance(msg, HumanMessage) else "assistant",
#                 "content": msg.content,
#             }
#             for msg in messages
#         ]
#         st.rerun()

# # --------- Main ---------
# for message in st.session_state["message_history"]:
#     with st.chat_message(message["role"]):
#         st.text(message["content"])

# user_input = st.chat_input("Type here")

# if user_input:
#     # User message
#     st.session_state["message_history"].append(
#         {"role": "user", "content": user_input}
#     )

#     with st.chat_message("user"):
#         st.text(user_input)

#     # CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
#     # This Config is for Langsmith
#     CONFIG = {
#         "configurable": {"thread_id": st.session_state["thread_id"]},
#         "metadata": {
#             "thread_id": st.session_state["thread_id"]
#         },
#         "run_name": "tool_bot",
#     }

#     # Assistant streaming response
#     # with st.chat_message("assistant"):

#     #     def ai_only_stream():
#     #         for message_chunk, _ in chatbot.stream(
#     #             {"messages": [HumanMessage(content=user_input)]},
#     #             config=CONFIG,
#     #             stream_mode="messages",
#     #         ):
#     #             if isinstance(message_chunk, AIMessage):
#     #                 yield message_chunk.content

#     #     ai_message = st.write_stream(ai_only_stream())

#     with st.chat_message("assistant"):
#         # Use a mutable holder so the generator can set/modify it
#         status_holder = {"box": None}

#         def ai_only_stream():
#             for message_chunk, metadata in chatbot.stream(
#                 {"messages": [HumanMessage(content=user_input)]},
#                 config=CONFIG,
#                 stream_mode="messages",
#             ):
#                 # Lazily create & update the SAME status container when any tool runs
#                 if isinstance(message_chunk, ToolMessage):
#                     tool_name = getattr(message_chunk, "name", "tool")
#                     if status_holder["box"] is None:
#                         status_holder["box"] = st.status(
#                             f"🔧 Using `{tool_name}` …", expanded=True
#                         )
#                     else:
#                         status_holder["box"].update(
#                             label=f"🔧 Using `{tool_name}` …",
#                             state="running",
#                             expanded=True,
#                         )

#                 # Stream ONLY assistant tokens
#                 if isinstance(message_chunk, AIMessage):
#                     yield message_chunk.content

#         ai_message = st.write_stream(ai_only_stream())

#         # Finalize only if a tool was actually used
#         if status_holder["box"] is not None:
#             status_holder["box"].update(
#                 label="✅ Tool finished", state="complete", expanded=False
#             )

#     st.session_state["message_history"].append(
#         {"role": "assistant", "content": ai_message}
#     )
#     st.rerun()