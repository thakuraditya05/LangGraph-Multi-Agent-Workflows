import streamlit as st
import uuid
import os
from tool_llm.langgraph_tool_backend import chatbot, retrieve_all_threads,tools
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def render_tool_llm_ui():
    # Make sure this imports from your actual backend file name


    # =========================== Helper: Clean Text Extractor ===========================
    def get_clean_text(content):
        """Extracts plain text from Gemini's list/dict response format."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Extracts 'text' value from dictionaries in the list
            text_chunks = [item['text'] for item in content if isinstance(item, dict) and 'text' in item]
            return "\n".join(text_chunks)
        return str(content)

    # =========================== Utilities ===========================
    def generate_thread_id():
        return str(uuid.uuid4())

    def reset_chat():
        thread_id = generate_thread_id()
        st.session_state["thread_id"] = thread_id
        add_thread(thread_id)
        st.session_state["message_history"] = []

    def add_thread(thread_id):
        if thread_id not in st.session_state["chat_threads"]:
            st.session_state["chat_threads"].append(thread_id)

    def load_conversation(thread_id):
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])

    # ======================= Session Initialization ===================
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = generate_thread_id()

    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = retrieve_all_threads()
        if st.session_state["thread_id"] not in st.session_state["chat_threads"]:
            add_thread(st.session_state["thread_id"])

    # ============================ Sidebar ============================
    st.sidebar.title("🧠 LangGraph AI Agent")
    st.sidebar.markdown("Equipped with Web, Finance, Social, and MongoDB Tools.")
    tool_names = ", ".join([f"`{t.name}`" for t in tools])
    st.sidebar.markdown(f"🛠️ **Active Agent Tools:** {tool_names}")

    # 
    if st.sidebar.button("➕ New Chat"):
        reset_chat()
        st.rerun()

    st.sidebar.header("History")
    for thread_id in st.session_state["chat_threads"][::-1]:
        if st.sidebar.button(f"Session: {str(thread_id)[:8]}...", key=thread_id):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    temp_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content:
                    # Apply text cleaner to history as well
                    clean_content = get_clean_text(msg.content)
                    if clean_content.strip():
                        temp_messages.append({"role": "assistant", "content": clean_content})
                elif isinstance(msg, ToolMessage):
                    temp_messages.append({"role": "tool", "name": msg.name, "content": msg.content})
            
            st.session_state["message_history"] = temp_messages
            st.rerun()

    # ============================ Main UI ============================
    st.title("AI Intelligence Hub 🚀")

    st.caption("Ask me to analyze stocks, scrape web, query MongoDB, or fetch trends!")

    # 1st Dropdown: Tool Information (English)
    with st.expander("🛠️ **Available Tools & Their Capabilities (Click to expand)**", expanded=False):
        st.markdown("""
        | Tool Name | Simple Explanation |
        | :--- | :--- |
        | **Web Scraper** | Provide any URL, and it will extract and read the main text content. |
        | **Tavily Search** | Performs real-time internet searches for live information and news. |
        | **Market Data** | Fetches live stock prices, market cap, and P/E ratios (e.g., TSLA, NVDA). |
        | **Reddit Trends** | Retrieves the top trending posts from any specific Reddit community. |
        | **YouTube Trends** | Searches and provides a list of top trending YouTube videos and their links. |
        | **MongoDB Tools** | Full database control: insert data, query records, and explore databases/collections. |
        | **Save Report** | Compiles research into a properly formatted Markdown (.md) file with a download button. |
        """)

    # 2nd Dropdown: Testing Prompts for ALL Tools
    with st.expander("🧪 **Testing Prompts (Copy & Paste to test each tool)**", expanded=False):
        st.markdown("""
        * 🕸️ **Web Scraper:** `Scrape the text from https://en.wikipedia.org/wiki/Artificial_intelligence and give me a 3-point summary.`
        * 🔍 **Tavily Search:** `Search the web using Tavily for the latest news on 'LangGraph AI agents' from this month and give me a quick summary.`
        * 📈 **Market Data:** `Fetch the current stock market data and P/E ratio for Nvidia (NVDA).`
        * 📱 **Reddit Trends:** `What are the top 5 trending posts on the 'MachineLearning' subreddit right now?`
        * 🎥 **YouTube Trends:** `Find the top trending YouTube videos about 'Python AI Agent Development'.`
        * 🗄️ **MongoDB Tools:** `List all the databases currently available on my MongoDB cluster, and then list the collections in the 'Growthify_db' database.`
        * 💾 **Save Report (Multi-Tool):** `Fetch the current stock market data for Apple (AAPL). Generate a neat executive summary and save the report to apple_stock.md.`
        """)
    # 1. Render existing history correctly
    for message in st.session_state["message_history"]:
        if message["role"] == "tool":
            with st.expander(f"🛠️ Tool Used: `{message.get('name', 'Unknown')}`", expanded=False):
                st.code(message["content"], language="json")
                
                if message.get("name") == "save_report_to_file":
                    content_str = message.get("content", "")
                    if "Success: Report saved locally as " in content_str:
                        try:
                            filename = content_str.split("as ")[1].split(" in")[0].strip()
                            if os.path.exists(filename):
                                with open(filename, "rb") as f:
                                    st.download_button(
                                        label=f"⬇️ Download {filename}",
                                        data=f,
                                        file_name=filename,
                                        mime="text/markdown",
                                        key=f"dl_{filename}_{uuid.uuid4()}"
                                    )
                        except:
                            pass
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 2. Handle new user input
    user_input = st.chat_input("Enter your request...")

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        CONFIG = {
            "configurable": {"thread_id": st.session_state["thread_id"]},
            "recursion_limit": 10 
        }

        with st.chat_message("assistant"):
            st_placeholder = st.empty()
            
            events = chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=CONFIG, 
                stream_mode="values"
            )
            
            final_ai_text = ""
            download_rendered = False
            
            for event in events:
                latest_msg = event["messages"][-1]
                
                if isinstance(latest_msg, ToolMessage):
                    with st.expander(f"🛠️ Tool Executed: `{latest_msg.name}`", expanded=True):
                        st.code(latest_msg.content, language="json")
                        
                        
                        # --- NEW: LIVE DOWNLOAD BUTTON LOGIC ---
                        if latest_msg.name == "save_report_to_file" and not download_rendered:
                            content_str = str(latest_msg.content)
                            if "Success: Report saved locally as " in content_str:
                                try:
                                    filename = content_str.split("as ")[1].split(" in")[0].strip()
                                    if os.path.exists(filename):
                                        with open(filename, "rb") as f:
                                            st.download_button(
                                                label=f"⬇️ Download {filename}",
                                                data=f,
                                                file_name=filename,
                                                mime="text/markdown",
                                                key=f"dl_live_{filename}"
                                            )
                                        download_rendered = True
                                except Exception as e:
                                    st.error(f"Could not render download button: {e}")
                                    
                                    
                    st.session_state["message_history"].append({
                        "role": "tool", 
                        "name": latest_msg.name, 
                        "content": latest_msg.content
                    })
                    
                elif isinstance(latest_msg, AIMessage) and latest_msg.content:
                    # Clean the raw content before displaying it
                    clean_text = get_clean_text(latest_msg.content)
                    
                    # Only update UI if there is actual text to show
                    if clean_text.strip():
                        final_ai_text = clean_text
                        st_placeholder.markdown(final_ai_text)
                    
            if final_ai_text:
                st.session_state["message_history"].append({"role": "assistant", "content": final_ai_text})
