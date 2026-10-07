import os
import re
import asyncio
import ast
import threading
import aiosqlite
import time
from typing import List, TypedDict, Annotated, Dict
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()
os.environ.setdefault("GEMINI_API_KEY_1", os.getenv("GEMINI_API_KEY", ""))
os.environ.setdefault(
    "GEMINI_API_KEY_2",
    os.getenv("GEMINI_API_KEY_TOOL") or os.getenv("GEMINI_API_KEY", ""),
)

# ==========================================
# 1.helper function 
# ==========================================

def clean_gemini_backend_text(content):
    if isinstance(content, str):
        content = content.strip()
        if (content.startswith("[{") and content.endswith("]")) or (content.startswith("{") and content.endswith("}")):
            try:
                parsed = ast.literal_eval(content)
                return clean_gemini_backend_text(parsed)
            except (ValueError, SyntaxError):
                return content
        return content
    elif isinstance(content, list):
        return "".join(clean_gemini_backend_text(item) for item in content)
    elif isinstance(content, dict):
        return content.get("text", "")
    return str(content)

# ==========================================
# 1. Background Async Loop
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    return _submit_async(coro)

# ==========================================
# 2. Gemini LLM & Embeddings Setup
# ==========================================


# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash-lite", 
#     temperature=0,
#     google_api_key=os.environ.get("GEMINI_API_KEY")
# )

primary_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    temperature=0, 
    max_retries=1,
    google_api_key=os.environ.get("GEMINI_API_KEY_1")
    )
fallback_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    temperature=0, 
    max_retries=2,
    google_api_key=os.environ.get("GEMINI_API_KEY_2")
)
llm = primary_llm.with_fallbacks([fallback_llm])



primary_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",  # <-- Yahan change kiya gaya hai
    google_api_key=os.environ.get("GEMINI_API_KEY")
)
fallback_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.environ.get("GEMINI_API_KEY_2"),
)

class FallbackEmbeddings(Embeddings):
    """Try the primary embedding provider, then the secondary Gemini key."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return primary_embeddings.embed_documents(texts)
        except Exception:
            return fallback_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            return primary_embeddings.embed_query(text)
        except Exception:
            return fallback_embeddings.embed_query(text)


embeddings = FallbackEmbeddings()

# ==========================================
# 3. Dynamic PDF Handling
# ==========================================
thread_retrievers: Dict[str, any] = {}

def process_pdf_for_thread(thread_id: str, file_path: str):
    try:
        docs = PyPDFLoader(file_path).load()
        # chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(docs)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150).split_documents(docs)
        for d in chunks:
            d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        thread_retrievers[thread_id] = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        return True
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return False

# ==========================================
# 4. State Structure
# ==========================================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] 
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str

# ==========================================
# 5. Core Nodes
# ==========================================
async def extract_and_contextualize_node(state: State) -> State:
    last_msg = state["messages"][-1].content
    history_messages = state["messages"][:-1][-4:] 
    
    if not history_messages:
        return {"question": last_msg}
    
    history_str = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in history_messages])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user question into a standalone question using the chat history context. DO NOT answer it."),
        ("human", "Chat History:\n{history_str}\n\nNew Question: {last_msg}")
    ])
    
    res = await (prompt | llm).ainvoke({
        "history_str": history_str, 
        "last_msg": last_msg
    })
    
    clean_question = clean_gemini_backend_text(res.content)
    
    return {"question": clean_question}

def retrieve_node(state: State, config: RunnableConfig) -> State:
    thread_id = config["configurable"]["thread_id"]
    q = state["question"]
    
    retriever = thread_retrievers.get(str(thread_id))
    if retriever:
        return {"docs": retriever.invoke(q)}
    return {"docs": []}

class DocEvalScore(BaseModel):
    score: float
    reason: str

doc_eval_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict retrieval evaluator for RAG.\n"
     "Return a relevance score in [0.0, 1.0].\n"
     "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
     "- 0.0: chunk is irrelevant\n"
     "Output JSON only."),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])
doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)

UPPER_TH = 0.7
LOWER_TH = 0.3

def eval_each_doc_node(state: State) -> State:
    q = state["question"]
    scores: List[float] = []
    good: List[Document] = []
    
    if not state["docs"]:
        return {"good_docs": [], "verdict": "INCORRECT", "reason": "No documents retrieved (No PDF)."}

    for d in state["docs"]:
        time.sleep(1.5)  # <--- ANTI-RATE LIMIT DELAY: Prevents 429 Exhausted Error
        out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
        scores.append(out.score)
        if out.score > LOWER_TH:
            good.append(d)

    if any(s > UPPER_TH for s in scores):
        return {"good_docs": good, "verdict": "CORRECT", "reason": f"Chunk scored > {UPPER_TH}."}
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {"good_docs": [], "verdict": "INCORRECT", "reason": f"All chunks scored < {LOWER_TH}."}
    return {"good_docs": good, "verdict": "AMBIGUOUS", "reason": "Mixed relevance scores."}

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

# <--- BATCH FILTERING TO PREVENT RATE LIMITS --->
class BatchFilter(BaseModel):
    kept_indices: List[int]

filter_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a context relevance filter. Given a user input and a numbered list of sentences.\n"
     "Return ONLY a JSON list of the INDICES of the sentences that relate to the user's input, topic, or intent.\n"
     "If the input is a single word, concept, or greeting, keep sentences that define or explain it.\n"
     "Return [] only if the sentence is completely unrelated."),
    ("human", "Input: {question}\n\nSentences:\n{sentences}"),
])
filter_chain = filter_prompt | llm.with_structured_output(BatchFilter)

def refine(state: State) -> State:
    q = state["question"]
    if state.get("verdict") == "CORRECT":
        docs_to_use = state["good_docs"]
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state["web_docs"]
    else:  
        docs_to_use = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    
    # 🚨 ANTI-CRASH: Agar context bohot bada hai, toh usko truncate (cut) kar do
    MAX_CHARS = 10000  # Approx 2500 tokens
    if len(context) > MAX_CHARS:
        context = context[:MAX_CHARS] + "... [TRUNCATED FOR SAFETY]"
    
    if not context:
        return {"strips": [], "kept_strips": [], "refined_context": ""}

    strips = decompose_to_sentences(context)
    
    # 🚨 ANTI-CRASH: API ko 30 se zyada sentences mat bhejo, warna Rate Limit hit hogi
    if len(strips) > 30:
        strips = strips[:30]
    
    # Send all sentences in ONE batched request
    sentences_text = "\n".join([f"[{i}] {s}" for i, s in enumerate(strips)])
    
    time.sleep(2) # Small delay to ensure we are safe
    try:
        res = filter_chain.invoke({"question": q, "sentences": sentences_text}, config={"callbacks": []})
        kept = [strips[i] for i in res.kept_indices if i < len(strips)]
    except Exception as e:
        print("Batch Filter Error:", e)
        kept = strips # Fallback to keeping everything if LLM fails formatting
        
    if not kept and strips:
        print("Filter removed everything! Falling back to original strips.")
        kept = strips  # Original sentences ko hi wapas rakh lo

    refined_context = "\n".join(kept).strip()
    return {"strips": strips, "kept_strips": kept, "refined_context": refined_context}

class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user question into a web search query (6-14 words). DO NOT answer it. Return JSON."),
    ("human", "Question: {question}"),
])
rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)

def rewrite_query_node(state: State) -> State:
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}

tavily = TavilySearchResults(max_results=5)
# tavily = TavilySearch(max_results=5)

# def web_search_node(state: State) -> State:
#     q = state.get("web_query") or state["question"]
#     results = tavily.invoke({"query": q})
#     web_docs: List[Document] = []
#     for r in results or []:
#         title = r.get("title", "")
#         url = r.get("url", "")
#         content = r.get("content", "") or r.get("snippet", "")
#         text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
#         web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
#     return {"web_docs": web_docs}
def web_search_node(state: State) -> State:
    q = state.get("web_query") or state["question"]
    results = tavily.invoke({"query": q})
    
    # 🚨 FIX: Agar results string format (JSON) me aaye, toh usko List me convert karein
    if isinstance(results, str):
        import json
        import ast
        try:
            results = json.loads(results)
        except Exception:
            try:
                results = ast.literal_eval(results)
            except Exception:
                results = []

    web_docs: List[Document] = []
    # Ab 'results' humesha list of dictionaries hoga
    for r in results or []:
        if isinstance(r, dict):  # Extra safety check
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
            
    return {"web_docs": web_docs}

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent AI. Formulate your response based ONLY on the provided context.\n"
               "- If the user input is a question, answer it directly using the context.\n"
               "- If the user input is a single word, concept, or greeting, use the context to explain its meaning, origin, or relevant facts.\n"
               "- If the context is entirely empty, only then say 'Insufficient context provided.'"),
    ("human", "Input: {question}\n\nContext:\n{context}"),
])

async def generate(state: State) -> State:
    out = await (answer_prompt | llm).ainvoke({"question": state["question"], "context": state["refined_context"]})
    return {"answer": out.content, "messages": [AIMessage(content=out.content)]}

def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    else:
        return "rewrite_query"

# ==========================================
# 6. Graph Compilation
# ==========================================
g = StateGraph(State)
g.add_node("extract_question", extract_and_contextualize_node)
g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_doc", eval_each_doc_node)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search", web_search_node)
g.add_node("refine", refine)
g.add_node("generate", generate)

g.add_edge(START, "extract_question")
g.add_edge("extract_question", "retrieve")
g.add_edge("retrieve", "eval_each_doc")
g.add_conditional_edges("eval_each_doc", route_after_eval, {"refine": "refine", "rewrite_query": "rewrite_query"})
g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)

async def _init_checkpointer():
    conn = await aiosqlite.connect(database="crag_chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())
chatbot = g.compile(checkpointer=checkpointer)

async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())









