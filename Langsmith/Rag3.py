# Here the whole project is now traced in once all things are traced in single system(with its own tags and metadata)
# But here still the issue of other proccesses rather then chaining is still there for every run which takes time in everyrun
import os
from dotenv import load_dotenv
from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGCHAIN_PROJECT"] = "Rag1 LLM"
load_dotenv()

PDF_PATH = "Attention.pdf" 

@traceable(name="load_pdf",tags=['Pdf',"loader"],metadata={"loader":"pypdfloader"})
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()  

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore",tags=['embedding','vectorstore'],metadata={"embedding_model":"nomic-embed-text"})
def build_vectorstore(splits):
    emb = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.from_documents(splits, emb)

# ----------------- parent setup function (traced) -----------------
@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=150):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits)
    return vs

llm = ChatOllama(model="llama3.2:3b", temperature=0.1)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer ONLY from the provided context. If not found, say you don't know.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ----------------- one top-level (root) run -----------------
@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(pdf_path: str, question: str):
    # Parent setup run (child of root)
    vectorstore = setup_pipeline(pdf_path, chunk_size=1000, chunk_overlap=150)

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    parallel = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    chain = parallel | prompt | llm | StrOutputParser()

    # This LangChain run stays under the same root (since we're inside this traced function)
    lc_config = {"run_name": "pdf_rag_query"}
    return chain.invoke(question, config=lc_config)

if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA:", ans)