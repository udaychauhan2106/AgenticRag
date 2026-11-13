import logging
import os
import time
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

AI_AVATAR = "🤖"
USER_AVATAR = "👤"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")

    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

st.title("🤖 Enterprise Agentic Assistant")

for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documentation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        with st.status("🔍 Agent is thinking...", expanded=True) as status:
            try:
                base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                response = requests.post(
                    f"{base_url}/query",
                    json={"q": prompt, "thread_id": st.session_state.session_id},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError):
                logger.exception("Could not complete the RAG request")
                status.update(label="❌ Request failed", state="error")
                st.error("Could not complete the request. Check the backend connection and logs.")
                st.stop()

            for step in data.get("thought_process", []):
                st.write(f"⚙️ {step}")

            status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

            sources = data.get("sources", [])
            if sources:
                with st.expander("📄 View Retrieved Context (Sources)"):
                    for index, source in enumerate(sources):
                        preview = source[:100].replace("\n", " ") + "..."
                        with st.expander(f"Chunk {index + 1}: {preview}"):
                            st.info(source)

        answer_placeholder = st.empty()
        full_answer = data.get("answer") or "No response."
        current_text = ""
        for character in full_answer:
            current_text += character
            answer_placeholder.markdown(current_text + "▌")
            time.sleep(0.005)

        answer_placeholder.markdown(full_answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer}
        )
