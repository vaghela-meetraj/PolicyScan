"""Streamlit Web UI for PolicyScan Insurance Assistant."""

import os
import streamlit as st
from insurance_assistant import config
from insurance_assistant.pipeline import build_insurance_assistant, ask

# Page setup
st.set_page_config(
    page_title="PolicyScan",
    page_icon="assets/assistant.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inline SVGs
SHIELD_SVG = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>"""
SPARKLE_SVG = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>"""
CONFIG_SVG = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>"""

# Pixel-perfect chat and container styling
st.markdown(
    """
    <style>
    /* Main layout constraint */
    .block-container {
        max-width: 860px;
        padding-top: 1.75rem;
        padding-bottom: 5rem;
    }

    /* Status badge */
    .status-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(34, 197, 94, 0.12);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.25);
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* Common message card structure */
    [data-testid="stChatMessage"] {
        padding: 0.85rem 1.15rem;
        margin-bottom: 0.85rem;
        border-radius: 14px;
        line-height: 1.55;
        font-size: 0.95rem;
    }

    /* User messages: anchored to the right */
    [data-testid="stChatMessage"]:has(img[src*="user.svg"]) {
        margin-left: 14%;
        margin-right: 0;
        flex-direction: row-reverse;
        background-color: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.22);
        border-bottom-right-radius: 3px;
    }

    /* Assistant messages: anchored to the left */
    [data-testid="stChatMessage"]:has(img[src*="assistant.svg"]) {
        margin-right: 14%;
        margin-left: 0;
        background-color: rgba(125, 125, 125, 0.06);
        border: 1px solid rgba(125, 125, 125, 0.18);
        border-bottom-left-radius: 3px;
    }

    /* Avatar styling */
    [data-testid="stChatMessageAvatar"] {
        margin-top: 2px;
    }
    [data-testid="stChatMessageAvatar"] img {
        border-radius: 50%;
    }

    /* Sidebar cards */
    .config-card {
        background: rgba(125, 125, 125, 0.06);
        border: 1px solid rgba(125, 125, 125, 0.14);
        border-radius: 10px;
        padding: 0.75rem 0.85rem;
        margin-top: 0.5rem;
    }
    .config-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        font-size: 0.82rem;
        border-bottom: 1px solid rgba(125, 125, 125, 0.08);
    }
    .config-item:last-child {
        border-bottom: none;
    }
    .config-item code {
        font-size: 0.8rem;
        padding: 2px 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load agent once across sessions
@st.cache_resource(show_spinner="Initializing policy assistant...")
def get_agent():
    return build_insurance_assistant()


# Initialize conversation state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am **PolicyScan**, your insurance policy assistant. "
                "Ask me any question about your policy coverage, limits, "
                "waiting periods, or claims procedures."
            ),
        }
    ]

# Sidebar navigation & settings
with st.sidebar:
    st.markdown(
        f"<h2 style='margin:0; padding:0; display:flex; align-items:center;'>{SHIELD_SVG}PolicyScan</h2>",
        unsafe_allow_html=True,
    )
    st.caption("AI-powered policy verification engine")

    st.write("")

    if st.button("New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am **PolicyScan**. How can I help you with your insurance policy today?",
            }
        ]
        st.rerun()

    st.write("")
    st.markdown(
        f"<div style='display:flex; align-items:center; font-weight:600; font-size:0.92rem; margin-bottom:8px;'>"
        f"{SPARKLE_SVG}Sample Questions</div>",
        unsafe_allow_html=True,
    )

    sample_questions = [
        "What is the waiting period for pre-existing diseases?",
        "What are the room rent and ICU charges limits?",
        "Is cataract surgery covered and what is the limit?",
        "What is co-payment and when does it apply?",
        "What is the procedure for intimating a claim?",
    ]

    for q in sample_questions:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            st.session_state["pending_input"] = q
            st.rerun()

    st.write("")
    st.markdown(
        f"<div style='display:flex; align-items:center; font-weight:600; font-size:0.92rem; margin-bottom:4px;'>"
        f"{CONFIG_SVG}Configuration</div>",
        unsafe_allow_html=True,
    )

    policy_filename = os.path.basename(config.data_path)
    st.markdown(
        f"""
        <div class="config-card">
            <div class="config-item">
                <span style="color:#888;">Document</span>
                <code>{policy_filename}</code>
            </div>
            <div class="config-item">
                <span style="color:#888;">LLM</span>
                <code>{config.llm_model_name}</code>
            </div>
            <div class="config-item">
                <span style="color:#888;">Embeddings</span>
                <code>{config.embedding_model_name}</code>
            </div>
            <div class="config-item">
                <span style="color:#888;">Top-k Results</span>
                <code>{config.top_k_results}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main chat view
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2px;">
        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.5px;">Policy Inquiry</h1>
        <span class="status-pill">Active</span>
    </div>
    <p style="margin: 0 0 1.25rem 0; color: #888888; font-size: 0.92rem;">
        Ask questions regarding terms, benefits, waiting periods, and exclusions.
    </p>
    """,
    unsafe_allow_html=True,
)

# Render conversation history
for msg in st.session_state.messages:
    avatar = "assets/user.svg" if msg["role"] == "user" else "assets/assistant.svg"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Handle user input from chat box or sidebar buttons
user_input = st.chat_input("Ask a question about your insurance policy...")
if "pending_input" in st.session_state and st.session_state["pending_input"]:
    user_input = st.session_state.pop("pending_input")

if user_input:
    # User message (right-anchored)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="assets/user.svg"):
        st.markdown(user_input)

    # Assistant message (left-anchored)
    with st.chat_message("assistant", avatar="assets/assistant.svg"):
        with st.spinner("Searching policy documents..."):
            try:
                agent = get_agent()
                response = ask(agent, user_input)
            except Exception as e:
                response = f"An error occurred: {e}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
