import streamlit as st

# ==========================================
# Page Configuration & Global State
# ==========================================
st.set_page_config(
    page_title="AI Architecture Hub",
    page_icon="🤖",
    layout="wide",
)

if "selected_app" not in st.session_state:
    st.session_state.selected_app = "Home"

# ==========================================
# Custom CSS for Beautiful UI
# ==========================================
st.markdown(
    """
    <style>
    .project-card {
        min-height: 140px;
        padding: 1.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 18px 18px 0px 0px;
        background: linear-gradient(145deg, rgba(79, 70, 229, 0.12), rgba(14, 165, 233, 0.06));
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .project-card h3 { 
        margin-top: 0; 
        color: #4da6ff;
    }
    .project-card p { 
        min-height: 2.5rem; 
        opacity: 0.85; 
        font-size: 15px;
    }
    /* Customize the button */
    div.stButton > button[kind="secondary"] {
        border-radius: 8px;
        font-weight: 600;
        min-height: 2.8rem;
        margin-top: 10px;
        margin-bottom: 20px;
        border: 1px solid #4da6ff;
    }
    /* Customize the expander */
    div[data-testid="stExpander"] {
        border-radius: 0px 0px 18px 18px;
        border-top: none;
        background-color: rgba(255, 255, 255, 0.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# Sidebar Navigation
# ==========================================
if st.session_state.selected_app != "Home":
    if st.sidebar.button("⬅️ Back to Dashboard", use_container_width=True):
        st.session_state.selected_app = "Home"
        st.rerun()

st.sidebar.title("🤖 AI Architecture Hub")
if st.session_state.selected_app == "Home":
    st.sidebar.caption("Choose an architecture to explore.")

# ==========================================
# Reusable Project Card Component
# ==========================================
def project_card(title, description, features_list, app_name, button_label="Launch Project 🚀"):
    # Render Card Header & Description
    st.markdown(
        f'<div class="project-card"><h3>{title}</h3><p>{description}</p></div>',
        unsafe_allow_html=True,
    )
    
    # Render Drop-down (Expander) for Features
    with st.expander("✨ View Key Architecture Features"):
        for feature in features_list:
            st.markdown(f"- {feature}")
            
    # Render Launch Button
    if st.button(button_label, key=f"open_{app_name}", use_container_width=True):
        st.session_state.selected_app = app_name
        st.rerun()

# ==========================================
# Main Routing & Dashboard Layout
# ==========================================
selected_app = st.session_state.selected_app

if selected_app == "Home":
    st.title("AI Systems & Architectures Hub")
    st.write("Explore advanced Agentic Workflows, stateful RAG systems, and tool-connected LLMs built with LangGraph.")
    st.markdown("---")

    # Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        crag_features = [
            "**Agentic LangGraph Workflow:** Mimics a researcher's logic for intelligent document retrieval.",
            "**Dynamic Relevance Grading:** Evaluates and filters out noisy or off-topic document chunks.",
            "**Web Search Fallback:** Automatically rewrites queries and fetches real-time data via Tavily.",
            "**Fault-Tolerant Engine:** Features Dual API keys and HuggingFace embedding fallbacks."
        ]
        project_card(
            title="📚 Corrective RAG (CRAG)",
            description="A fault-tolerant retrieval system with relevance grading and automatic web search fallback.",
            features_list=crag_features,
            app_name="Corrective RAG"
        )
        
    with col2:
        self_rag_features = [
            "**Self-Reflective State Machine:** Dynamically critiques and refines its own generated answers.",
            "**Hallucination Grader (`is_sup`):** Strictly verifies that all claims are 100% supported by the context.",
            "**Usefulness Validator (`is_use`):** Ensures the final response actually solves the user's core intent.",
            "**Semantic Routing:** Intelligently routes queries between local FAISS indexing and external tools."
        ]
        project_card(
            title="🔎 Self-RAG Intelligence",
            description="An advanced system that evaluates its own answers, checks for hallucinations, and revises outputs.",
            features_list=self_rag_features,
            app_name="Self-RAG"
        )

    # Row 2
    col3, col4 = st.columns(2)
    
    with col3:
        tool_llm_features = [
            "**7 Integrated APIs:** Orchestrates Tavily, yfinance, Reddit (PRAW), YouTube, and Web Scraping in real-time.",
            "**Dynamic MongoDB Master:** Executes natural language CRUD operations across any database cluster.",
            "**Persistent Memory:** Utilizes SQLite Checkpointing for stateful cross-thread conversations.",
            "**Report Generator:** Synthesizes research into Markdown files with native Streamlit UI downloads."
        ]
        project_card(
            title="🛠️ Multi-Tool AI Agent",
            description="A stateful orchestrator capable of running web, finance, social media, and database tools.",
            features_list=tool_llm_features,
            app_name="Tool-Connected LLM"
        )
        
    with col4:
        project_4_features = [
            "Currently in development phase.",
            "Will feature multi-agent collaboration frameworks.",
            "More updates to be deployed soon."
        ]
        project_card(
            title="✨ Project 4 (Coming Soon)",
            description="A new experimental AI architecture will be deployed here in the future.",
            features_list=project_4_features,
            app_name="Project 4",
            button_label="View Status ⏳"
        )

# ==========================================
# Application Routes / Imports
# ==========================================
elif selected_app == "Corrective RAG":
    from corrective_rag.crag_frontend import render_crag_ui
    render_crag_ui()

elif selected_app == "Self-RAG":
    from self_rag.langgraph_frontend import render_self_rag_ui
    render_self_rag_ui()

elif selected_app == "Tool-Connected LLM":
    from tool_llm.streamlit_frontend_tool import render_tool_llm_ui
    render_tool_llm_ui()

elif selected_app == "Project 4":
    st.title("✨ Project 4")
    st.info("The Multi-Agent Collaboration system is currently under construction. Please check back later!")









# import streamlit as st


# st.set_page_config(
#     page_title="AI Architecture Hub",
#     page_icon="🤖",
#     layout="wide",
# )

# if "selected_app" not in st.session_state:
#     st.session_state.selected_app = "Home"

# st.markdown(
#     """
#     <style>
#     .project-card {
#         min-height: 245px;
#         padding: 1.5rem;
#         margin-bottom: 1rem;
#         border: 1px solid rgba(128, 128, 128, 0.3);
#         border-radius: 18px;
#         background: linear-gradient(145deg, rgba(79, 70, 229, 0.12), rgba(14, 165, 233, 0.06));
#         box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
#     }
#     .project-card h3 { margin-top: 0; }
#     .project-card p { min-height: 3.5rem; opacity: 0.82; }
#     div.stButton > button[kind="secondary"] {
#         border-radius: 10px;
#         font-weight: 600;
#         min-height: 2.8rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# if st.session_state.selected_app != "Home":
#     if st.sidebar.button("⬅️ Back to Dashboard", use_container_width=True):
#         st.session_state.selected_app = "Home"
#         st.rerun()

# st.sidebar.title("🤖 AI Architecture Hub")
# if st.session_state.selected_app == "Home":
#     st.sidebar.caption("Choose an architecture to explore.")


# def project_card(title, description, app_name, button_label="Open project"):
#     st.markdown(
#         f'<div class="project-card"><h3>{title}</h3><p>{description}</p></div>',
#         unsafe_allow_html=True,
#     )
#     if st.button(button_label, key=f"open_{app_name}", use_container_width=True):
#         st.session_state.selected_app = app_name
#         st.rerun()


# selected_app = st.session_state.selected_app
# if selected_app == "Home":
#     st.title("AI Architecture Hub")
#     st.write("Explore the RAG and tool-connected AI projects in this workspace.")

#     col1, col2 = st.columns(2)
#     with col1:
#         project_card(
#             "📚 Corrective RAG",
#             "Retrieval with relevance grading and web search fallback.",
#             "Corrective RAG",
#         )
#     with col2:
#         project_card(
#             "🔎 Self-RAG",
#             "Retrieval, answer support checks, and response revision.",
#             "Self-RAG",
#         )

#     col3, col4 = st.columns(2)
#     with col3:
#         project_card(
#             "🛠️ Tool-Connected LLM",
#             "An agent that can call web, finance, social, and database tools.",
#             "Tool-Connected LLM",
#         )
#     with col4:
#         project_card(
#             "✨ Project 4",
#             "A new AI architecture will be available here soon.",
#             "Project 4",
#             button_label="View status",
#         )
# elif selected_app == "Corrective RAG":
#     from corrective_rag.crag_frontend import render_crag_ui

#     render_crag_ui()
# elif selected_app == "Self-RAG":
#     from self_rag.langraph_frontend import render_self_rag_ui

#     render_self_rag_ui()
# elif selected_app == "Tool-Connected LLM":
#     from tool_llm.streamlit_frontend_tool import render_tool_llm_ui

#     render_tool_llm_ui()
# elif selected_app == "Project 4":
#     st.title("Project 4")
#     st.info("Project 4 coming soon")
