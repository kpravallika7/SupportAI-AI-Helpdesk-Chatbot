"""
SupportAI — AI Helpdesk Chatbot
Streamlit web interface for the existing SupportAI helpdesk engine.

The existing supportai.py remains the core AI/retrieval layer.
This file provides the browser-based presentation layer.
"""

import streamlit as st

from supportai import (
    FAQMatcher,
    LLMClient,
    SupportAgent,
    faqs,
    GROQ_API_KEY,
    GROQ_MODEL,
)


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="SupportAI | AI Helpdesk Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Custom UI styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: #f7f9fc;
            color: #111827;
        }

        [data-testid="stSidebar"] {
            background: #111827;
        }

        [data-testid="stSidebar"] * {
            color: #f9fafb;
        }

        .brand {
            padding: 8px 0 18px 0;
        }

        .brand-title {
            font-size: 1.55rem;
            font-weight: 750;
            margin: 0;
            color: #111827;
        }

        .brand-subtitle {
            color: #6b7280;
            margin-top: 4px;
            font-size: 0.92rem;
        }

        .hero {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 24px 28px;
            margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }

        .hero h1 {
            margin: 0;
            color: #111827;
            font-size: 2rem;
        }

        .hero p {
            color: #6b7280;
            margin: 8px 0 0 0;
            font-size: 1rem;
        }

        .status-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 14px 16px;
            margin: 10px 0 18px 0;
        }

        .status-label {
            font-size: 0.78rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .status-value {
            font-weight: 650;
            color: #111827;
            margin-top: 2px;
        }

        .example-title {
            color: #374151;
            font-weight: 650;
            margin-bottom: 8px;
        }

        .footer {
            text-align: center;
            color: #9ca3af;
            font-size: 0.78rem;
            padding: 24px 0 8px 0;
        }

        /* Chat messages */
        div[data-testid="stChatMessage"] {
            border-radius: 14px;
        }

        /* Fix assistant/user message text visibility.
           Streamlit can inherit a light/dark theme text color that
           becomes nearly invisible on our light chat background. */
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] em {
            color: #111827 !important;
            opacity: 1 !important;
        }

        /* Make the assistant response area clearly readable. */
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            background-color: transparent !important;
        }

        /* Keep confidence/details text readable too. */
        div[data-testid="stChatMessage"] [data-testid="stCaptionContainer"],
        div[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] * {
            color: #6b7280 !important;
            opacity: 1 !important;
        }

        /* Chat input text and placeholder */
        /* Chat input: dark background, light readable text */
        div[data-testid="stChatInput"] {
            background-color: #24252f !important;
        }

        div[data-testid="stChatInput"] textarea {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            caret-color: #ffffff !important;
            background-color: transparent !important;
        }

        div[data-testid="stChatInput"] textarea::placeholder {
            color: #aeb4c0 !important;
            -webkit-text-fill-color: #aeb4c0 !important;
            opacity: 1 !important;
        }

        /* Example/action buttons: dark background with white text */
        div[data-testid="stButton"] > button {
            background-color: #111827 !important;
            color: #ffffff !important;
            border-color: #374151 !important;
        }

        div[data-testid="stButton"] > button p,
        div[data-testid="stButton"] > button span,
        div[data-testid="stButton"] > button div {
            color: #ffffff !important;
            opacity: 1 !important;
        }

        div[data-testid="stButton"] > button:hover {
            background-color: #1f2937 !important;
            color: #ffffff !important;
        }

        div[data-testid="stButton"] > button:hover p,
        div[data-testid="stButton"] > button:hover span,
        div[data-testid="stButton"] > button:hover div {
            color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Session-state helpers
# ------------------------------------------------------------
def create_agent():
    """Create a fresh SupportAI agent with the existing retrieval engine."""
    matcher = FAQMatcher(faqs)
    llm = LLMClient(GROQ_API_KEY, GROQ_MODEL)
    return SupportAgent(faqs, llm, matcher)


def initialize_session():
    """Initialize browser-session state."""
    if "agent" not in st.session_state:
        st.session_state.agent = create_agent()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm SupportAI, your AI helpdesk assistant. "
                    "I can help with account, billing, shipping, and "
                    "technical-support questions."
                ),
                "confidence": None,
                "faq_id": None,
            }
        ]


def reset_chat():
    """Start a new conversation."""
    st.session_state.agent = create_agent()
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "New conversation started. How can I help you today?"
            ),
            "confidence": None,
            "faq_id": None,
        }
    ]


initialize_session()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div style="font-size:2rem;">🤖</div>
            <div style="font-size:1.35rem;font-weight:750;">SupportAI</div>
            <div style="opacity:0.75;font-size:0.85rem;">
                AI Helpdesk Assistant
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.markdown("---")

    st.markdown("### Quick actions")

    if st.button("🧹 Clear Conversation", use_container_width=True):
        reset_chat()
        st.rerun()

    if st.button("👨‍💼 Escalate to Human", use_container_width=True):
        try:
            response = st.session_state.agent.escalate()
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "confidence": None,
                    "faq_id": None,
                }
            )
        except Exception as exc:
            st.error(f"Unable to escalate the request: {exc}")

    st.markdown("---")

    st.markdown("### Supported topics")
    st.markdown(
        """
        - 🔐 Account & password
        - 💳 Billing & refunds
        - 📦 Shipping & delivery
        - ✉️ Account email
        - 🛠️ Technical issues
        """
    )

    st.markdown("---")

    st.caption("Powered by Python • Scikit-learn • Groq • Streamlit")


# ------------------------------------------------------------
# Main header
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🤖 SupportAI</h1>
        <p>
            AI-powered customer support using FAQ retrieval,
            hybrid search, confidence checks, and LLM-generated responses.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Status
st.markdown(
    """
    <div class="status-card">
        <div class="status-label">Assistant status</div>
        <div class="status-value">● Online & ready to help</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Example prompts
# ------------------------------------------------------------
st.markdown('<div class="example-title">Try an example</div>', unsafe_allow_html=True)

example_cols = st.columns(4)
example_prompts = [
    "How do I reset my password?",
    "What is your refund policy?",
    "How long does shipping take?",
    "Why does the app keep crashing?",
]

for col, prompt in zip(example_cols, example_prompts):
    if col.button(prompt, use_container_width=True):
        st.session_state.pending_prompt = prompt
        st.rerun()


# ------------------------------------------------------------
# Chat history
# ------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(
        message["role"],
        avatar="🤖" if message["role"] == "assistant" else "👤",
    ):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("confidence") is not None
        ):
            confidence = float(message["confidence"])
            faq_id = message.get("faq_id")

            details = f"Confidence: {confidence:.2f}"
            if faq_id:
                details += f" • FAQ: {faq_id}"

            st.caption(details)


# ------------------------------------------------------------
# Input handling
# ------------------------------------------------------------
pending_prompt = st.session_state.pop("pending_prompt", None)

user_prompt = st.chat_input(
    "Ask SupportAI a question...",
)

prompt = pending_prompt or user_prompt

if prompt:
    prompt = prompt.strip()

    if prompt:
        # Display/store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "confidence": None,
                "faq_id": None,
            }
        )

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate/store assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("SupportAI is thinking..."):
                try:
                    response = st.session_state.agent.handle_message(prompt)
                    turn = st.session_state.agent.history[-1]

                    confidence = turn.confidence
                    faq_id = turn.faq_id

                    st.markdown(response)

                    if confidence is not None:
                        details = f"Confidence: {float(confidence):.2f}"
                        if faq_id:
                            details += f" • FAQ: {faq_id}"
                        st.caption(details)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "confidence": confidence,
                            "faq_id": faq_id,
                        }
                    )

                except Exception as exc:
                    error_message = (
                        "I couldn't process that request right now. "
                        "Please check the application configuration "
                        "and your Groq API connection."
                    )
                    st.error(error_message)
                    st.caption(f"Technical detail: {exc}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "confidence": None,
                            "faq_id": None,
                        }
                    )


# ------------------------------------------------------------
# Conversation details
# ------------------------------------------------------------
with st.expander("🔎 Conversation details"):
    history = st.session_state.agent.history

    if not history:
        st.write("No conversation turns yet.")
    else:
        for index, turn in enumerate(history, start=1):
            role = "User" if turn.role == "user" else "SupportAI"
            st.write(f"**{index}. {role}:** {turn.content}")

            if turn.faq_id:
                st.caption(
                    f"FAQ: {turn.faq_id} • Confidence: {turn.confidence:.2f}"
                )


st.markdown(
    '<div class="footer">SupportAI • AI-powered helpdesk demonstration</div>',
    unsafe_allow_html=True,
)
