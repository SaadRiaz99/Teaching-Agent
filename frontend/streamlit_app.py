from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="SMIT AI Teaching Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .source-box {
        background-color: #e8f0fe;
        border-left: 4px solid #4a90d9;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .grounded-true {
        color: #0a0;
        font-weight: 600;
    }
    .grounded-false {
        color: #a00;
        font-weight: 600;
    }
    div[data-testid="stSidebarNav"] {display: none;}
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = set()


def call_api(endpoint: str, method: str = "POST", **kwargs) -> dict[str, Any] | None:
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    try:
        if method == "POST":
            resp = requests.post(url, **kwargs, timeout=120)
        else:
            resp = requests.get(url, **kwargs, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}. Ensure the backend is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = "Unknown error"
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"API error: {detail}")
        return None


def render_chat_tab():
    st.markdown('<p class="main-header">💬 Chat with SMIT Teaching Agent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about your uploaded SMIT course materials</p>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 View Sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-box">'
                            f"<strong>Source:</strong> {src['document']} | "
                            f"<strong>Relevance:</strong> {src['score']:.2f}<br>"
                            f"{src['content'][:200]}..."
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    if prompt := st.chat_input("Ask a question about your learning materials..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = call_api(
                    "chat",
                    json={
                        "message": prompt,
                        "conversation_id": st.session_state.conversation_id,
                    },
                )

            if result:
                answer = result.get("answer", "No response generated.")
                grounded = result.get("grounded", False)
                sources = result.get("sources", [])

                grounded_label = "✅ Grounded in sources" if grounded else "⚠️ Not found in sources"
                grounded_class = "grounded-true" if grounded else "grounded-false"
                st.markdown(
                    f'<p class="{grounded_class}">{grounded_label}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(answer)

                if sources:
                    with st.expander("📚 Source Citations", expanded=True):
                        for src in sources:
                            st.markdown(
                                f'<div class="source-box">'
                                f"<strong>📄 {src['document']}</strong> "
                                f"(score: {src['score']:.2f})<br>"
                                f"{src['content']}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()


def render_upload_tab():
    st.markdown('<p class="main-header">📤 Upload Learning Materials</p>', unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Upload PDF, DOCX, TXT, or Markdown files to build your knowledge base</p>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Max file size: 50MB per file",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.uploaded_docs:
                st.info(f"⏭️ '{uploaded_file.name}' already uploaded")
                continue

            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                result = call_api("upload", files=files)

            if result:
                st.session_state.uploaded_docs.add(uploaded_file.name)
                st.success(
                    f"✅ {result.get('message', 'Uploaded')} "
                    f"({result.get('chunks_count', 0)} chunks)"
                )

    if st.session_state.uploaded_docs:
        st.markdown("### 📚 Uploaded Documents")
        for doc in sorted(st.session_state.uploaded_docs):
            st.markdown(f"- {doc}")


def render_quiz_tab():
    st.markdown('<p class="main-header">📝 Generate Quiz</p>', unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Create practice quizzes from your uploaded learning materials</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        num_q = st.number_input("Number of questions", min_value=3, max_value=20, value=5)
    with col2:
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])
    with col3:
        topic = st.text_input("Topic (optional)", placeholder="e.g., Python variables")

    if st.button("🎯 Generate Quiz", type="primary", use_container_width=True):
        with st.spinner("Generating quiz..."):
            payload = {
                "num_questions": num_q,
                "difficulty": difficulty,
            }
            if topic:
                payload["topic"] = topic
            result = call_api("quiz", json=payload)

        if result and result.get("questions"):
            st.success(f"✅ Generated {result['total_questions']} questions")
            for i, q in enumerate(result["questions"], 1):
                with st.container():
                    st.markdown(f"### Q{i}: {q['question']}")
                    for j, opt in enumerate(q["options"]):
                        st.markdown(f"{chr(65+j)}. {opt}")

                    with st.expander("💡 Show Answer"):
                        st.markdown(f"**Correct answer:** {q['correct_answer']}")
                        st.markdown(f"**Explanation:** {q['explanation']}")
                    st.divider()
        else:
            st.warning("No questions could be generated. Upload more documents first.")


def render_summary_tab():
    st.markdown('<p class="main-header">📋 Generate Summary</p>', unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Get concise summaries of your uploaded course materials</p>",
        unsafe_allow_html=True,
    )

    max_length = st.slider("Summary length (words)", 100, 1000, 500)

    if st.button("📄 Generate Summary", type="primary", use_container_width=True):
        with st.spinner("Generating summary..."):
            result = call_api("summary", json={"document_ids": [], "max_length": max_length})

        if result:
            st.markdown("### Summary")
            st.markdown(result["summary"])


def render_recommendations_tab():
    st.markdown('<p class="main-header">🎯 Learning Recommendations</p>', unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-header'>Get personalized study recommendations based on your questions</p>",
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "What are you trying to learn?",
        placeholder="e.g., I want to understand machine learning basics...",
        height=100,
    )

    if st.button("🔍 Get Recommendations", type="primary", use_container_width=True) and question:
        with st.spinner("Analyzing..."):
            result = call_api(
                f"recommendations?question={requests.utils.quote(question)}&top_k=6",
                method="GET",
            )

        if result:
            st.markdown("### 📖 Recommended Next Steps")
            st.markdown(result["recommendations"])


def render_stats():
    result = call_api("stats", method="GET")
    if result:
        st.metric("Total Documents in Vector Store", result.get("total_documents", 0))


def main():
    init_session_state()

    st.sidebar.markdown("# 🎓 SMIT AI Agent")
    st.sidebar.markdown("---")

    status = call_api("health", method="GET")
    if status:
        st.sidebar.success("✅ Backend Connected")
        st.sidebar.caption(f"v{status.get('version', '?')}")
    else:
        st.sidebar.error("❌ Backend Disconnected")

    if st.sidebar.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()

    render_stats()
    st.sidebar.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Chat", "📤 Upload", "📝 Quiz", "📋 Summary"]
    )

    with tab1:
        render_chat_tab()
    with tab2:
        render_upload_tab()
    with tab3:
        render_quiz_tab()
    with tab4:
        render_summary_tab()

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Upload SMIT course materials, then ask questions. "
        "The agent will answer based only on your uploaded content."
    )


if __name__ == "__main__":
    main()
