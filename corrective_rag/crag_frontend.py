import os
import queue
import uuid
import streamlit as st
import tempfile
import ast
from corrective_rag.crag_backend import chatbot, retrieve_all_threads, submit_async_task, process_pdf_for_thread
from langchain_core.messages import AIMessage, HumanMessage


def render_crag_ui():
    # Page config sabse pehle aani chahiye







    def clean_gemini_text(content):
        """Gemini ke raw list/dict format se clean text nikalne ke liye."""
        if isinstance(content, str):
            content = content.strip()
            if content.startswith("[{") and content.endswith("]"):
                try:
                    parsed = ast.literal_eval(content)
                    return clean_gemini_text(parsed)
                except (ValueError, SyntaxError):
                    return content
            return content
        elif isinstance(content, list):
            return "".join(clean_gemini_text(item) for item in content)
        elif isinstance(content, dict):
            return content.get("text", "")
        return str(content)

    # =========================== Utilities ===========================
    def generate_thread_id():
        return str(uuid.uuid4())

    def reset_chat():
        thread_id = generate_thread_id()
        st.session_state["thread_id"] = thread_id
        if thread_id not in st.session_state["chat_threads"]:
            st.session_state["chat_threads"].append(thread_id)
        st.session_state["message_history"] = []
        st.session_state["pdf_uploaded"] = False 

    def load_conversation(thread_id):
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])

    # ======================= Session Init ===================
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = retrieve_all_threads()

    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []

    if "pdf_uploaded" not in st.session_state:
        st.session_state["pdf_uploaded"] = False

    if "thread_id" not in st.session_state:
        reset_chat()

    # ============================ Sidebar ============================
    st.error("🚨 **WARNING:** Upload PDFs with a maximum of 2 to 3 pages. This project uses Gemini's free embedding tier which has strict rate limits. Larger files will crash the server!")
    st.sidebar.title("🧠 CRAG Agent")
    st.sidebar.markdown("Equipped with Vector Retrieval, Grading, and Web Search Fallback.")

    if st.sidebar.button("➕ New Chat"):
        reset_chat()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("📂 Upload Knowledge Base")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF for RAG", type=["pdf"])

    if uploaded_file is None:
        st.session_state["pdf_uploaded"] = False

    elif uploaded_file is not None and not st.session_state.get("pdf_uploaded"):
        with st.spinner("Chunking & Vectorizing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            success = process_pdf_for_thread(str(st.session_state["thread_id"]), tmp_path)
            if success:
                st.session_state["pdf_uploaded"] = True
                st.sidebar.success("✅ PDF Indexed Successfully!")
            else:
                st.sidebar.error("❌ Failed to process PDF.")
            os.remove(tmp_path)

    if st.session_state.get("pdf_uploaded"):
        st.sidebar.success("📄 PDF is currently loaded in memory.")

    st.sidebar.header("History")
    for thread_id in st.session_state["chat_threads"][::-1]:
        if st.sidebar.button(f"Session: {str(thread_id)[:8]}...", key=thread_id):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)
            temp_messages = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                temp_messages.append({"role": role, "content": msg.content})
            st.session_state["message_history"] = temp_messages
            st.session_state["pdf_uploaded"] = False 
            st.rerun()

    # ============================ Main UI ============================
    st.title("Corrective RAG Intelligence Hub 🚀")
    st.caption("Ask me anything! I will retrieve documents, grade their relevance, filter context, and fallback to web search if necessary.")

    # 1st Dropdown: Node Explanations
    with st.expander("🛠 **CRAG Node Explanations (Click to expand)**", expanded=False):
        st.markdown("""
        * **`extract_question`**: Converts follow-up questions into standalone queries using chat history.
        * **`retrieve`**: Fetches semantically similar chunks from the FAISS Vector Database.
        * **`eval_each_doc`**: Grader that evaluates if retrieved chunks are Correct, Incorrect, or Ambiguous.
        * **`rewrite_query`**: If documents are incorrect, rewrites the query for a web search engine.
        * **`web_search`**: Uses Tavily API to fetch live information from the internet.
        * **`refine`**: Filters out useless sentences from the combined context to save tokens and improve accuracy.
        * **`generate`**: The generator that drafts a final answer based *only* on the refined context.
        """)

    # 2nd Dropdown: Dynamic Architecture Graph
    with st.expander("🕸️ **CRAG Architecture Graph**", expanded=False):
        st.caption("This is the actual LangGraph architecture running under the hood.")
        try:
            config = {
                "theme": "base",
                "themeVariables": {
                    "primaryTextColor": "#000000",  # <-- Isko #FFFFFF se #000000 (Black) kar dein
                    "primaryColor": "#E6E6FA",      # Box ka background color thoda light purplish
                    "edgeLabelBackground": "#333333",
                    "tertiaryTextColor": "#000000"  # Isko bhi black kar dein
                }
            }
            mermaid_code = chatbot.get_graph().draw_mermaid(frontmatter_config={"config": config})
            st.markdown(f"```mermaid\n{mermaid_code}\n```")
        except Exception as e:
            st.warning("Could not render the graph visually.")
            st.code(str(e))

    # 1. Render existing history correctly
    for message in st.session_state["message_history"]:
        if message["role"] == "tool":
            with st.expander(f"🛠️ Node Executed: `{message.get('name', 'Unknown')}`", expanded=False):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(clean_gemini_text(message["content"]))

    # 2. Handle new user input
    user_input = st.chat_input("Enter your query...")

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)

        CONFIG = {
            "configurable": {"thread_id": str(st.session_state["thread_id"])},
            "metadata": {"thread_id": str(st.session_state["thread_id"])},
        }

        with st.chat_message("assistant"):
            st_placeholder = st.empty()
            
            def run_crag_pipeline():
                event_queue = queue.Queue()
                
                async def generate_updates():
                    try:
                        async for event in chatbot.astream({"messages": [HumanMessage(content=user_input)]}, config=CONFIG, stream_mode="updates"):
                            event_queue.put(event)
                    except Exception as exc:
                        event_queue.put(("error", exc))
                    finally:
                        event_queue.put(None)

                submit_async_task(generate_updates())
                final_output = ""

                # Consume Queue and Build UI dynamically
                while True:
                    item = event_queue.get()
                    if item is None: break
                    
                    if isinstance(item, tuple) and item[0] == "error":
                        st.error(f"Execution Error: {item[1]}")
                        break

                    for node_name, state_update in item.items():
                        # Format node state for history storage
                        summary_parts = []
                        for k, v in state_update.items():
                            if k not in ["messages", "docs", "good_docs", "web_docs"]:
                                summary_parts.append(f"**{k}**: {v}")
                        text_content = " | ".join(summary_parts)
                        
                        st.session_state["message_history"].append({
                            "role": "tool", 
                            "name": node_name, 
                            "content": text_content
                        })
                        
                        # Live rendering of the node inside an expander
                        with st.expander(f"🛠️ Node Executed: `{node_name}`", expanded=True):
                            for k, v in state_update.items():
                                # Skip raw objects to keep UI clean
                                if k in ["messages", "docs", "good_docs", "web_docs"]:
                                    continue
                                    
                                if k == "verdict":
                                    color = "green" if v == "CORRECT" else "red" if v == "INCORRECT" else "orange"
                                    st.markdown(f"**{k}**: :{color}[{v}]")
                                elif isinstance(v, list):
                                    st.markdown(f"**{k}**:")
                                    for item_val in v:
                                        st.markdown(f"- {item_val}")
                                else:
                                    st.markdown(f"**{k}**: {v}")
                                    
                        # If this node produced an answer, update the placeholder
                        if node_name == "generate" and "answer" in state_update:
                            final_output = clean_gemini_text(state_update["answer"])
                            st_placeholder.markdown(final_output)

                return final_output

            ai_response = run_crag_pipeline()
            
            if ai_response:
                st.session_state["message_history"].append({"role": "assistant", "content": ai_response})
