from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Sequence

from dotenv import load_dotenv

try:
    import streamlit as st
except ImportError:  # pragma: no cover - handled at runtime when the app is launched.
    st = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

SYSTEM_PROMPT = (
    "You are a helpful, real-time conversational assistant. Answer the user's actual question "
    "directly, naturally, and accurately. Use prior conversation for context. If you are unsure, "
    "say so rather than inventing facts."
)
WELCOME_MESSAGE = (
    "Hi! I’m your AI assistant. Ask me anything to start a conversation."
)


class ChatbotError(RuntimeError):
    """A user-facing error raised when the chatbot cannot get a model response."""


def _create_client(provider: str, api_key: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ChatbotError(
            "The chatbot client SDK is not installed. Run: pip install -e \".[chat]\""
        ) from exc
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        return OpenAI(base_url=base_url, api_key="ollama")
    return OpenAI(api_key=api_key)


def _provider_error_message(error: Exception, provider: str) -> str:
    from openai import APIConnectionError, AuthenticationError, NotFoundError, RateLimitError

    provider_name = "Ollama" if provider == "ollama" else "OpenAI"
    if isinstance(error, AuthenticationError):
        return f"{provider_name} rejected the API credentials. Check the chatbot configuration."
    if isinstance(error, RateLimitError):
        if provider == "ollama":
            return "The local model is busy or unavailable. Wait a moment and try again."
        return (
            "OpenAI could not complete this reply because the API quota is exhausted or the "
            "request was rate-limited. Check your OpenAI billing and usage limits, then try again. "
            "For local chat without API credits, install Ollama and set CHAT_PROVIDER=ollama in .env."
        )
    if isinstance(error, NotFoundError):
        if provider == "ollama":
            model = os.getenv("OLLAMA_MODEL", "llama3.2")
            return f"Ollama model '{model}' was not found. Download it with: ollama pull {model}"
        return "The configured OpenAI model was not found. Check the OPENAI_MODEL setting in .env."
    if isinstance(error, APIConnectionError):
        if provider == "ollama":
            return "Cannot connect to Ollama. Start Ollama and make sure its local server is running."
        return "Could not connect to OpenAI. Check your internet connection and try again."
    return f"{provider_name} could not complete the request. Check the service and try again."


def stream_chat_reply(
    message: str,
    history: Sequence[tuple[str, str]] | None = None,
) -> Iterator[str]:
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("Message must not be empty.")

    provider = os.getenv("CHAT_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "ollama"}:
        raise ChatbotError("CHAT_PROVIDER must be either 'openai' or 'ollama'.")
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if provider == "openai" and (
        not api_key
        or api_key.lower()
        in {"replace-me", "your-openai-api-key", "repalce-me"}
    ):
        raise ChatbotError(
            f"No OpenAI API key is configured. Add OPENAI_API_KEY to "
            f"{PROJECT_ROOT / '.env'} or set CHAT_PROVIDER=ollama to use a local model."
        )

    client = _create_client(provider, api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in history or []:
        if role in {"user", "assistant"} and content and content.strip():
            messages.append({"role": role, "content": content.strip()})
    if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != cleaned:
        messages.append({"role": "user", "content": cleaned})

    try:
        stream = client.chat.completions.create(
            model=(
                os.getenv("OLLAMA_MODEL", "llama3.2")
                if provider == "ollama"
                else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            ),
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            stream=True,
        )
        for event in stream:
            content = event.choices[0].delta.content
            if content:
                yield content
    except ChatbotError:
        raise
    except Exception as exc:
        try:
            from openai import OpenAIError
        except ImportError:
            raise
        if isinstance(exc, OpenAIError):
            raise ChatbotError(_provider_error_message(exc, provider)) from exc
        raise ChatbotError("An unexpected error occurred while generating the reply.") from exc


def build_chatbot_reply(
    message: str,
    history: Sequence[tuple[str, str]] | None = None,
) -> str:
    return "".join(stream_chat_reply(message, history)).strip()


def main(argv: list[str] | None = None) -> int:
    if st is None:
        raise RuntimeError("Install the chat extra with: pip install -e '.[chat]' to run the Streamlit app.")

    st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; color: #f5f7fa; }
        div[data-testid="stChatMessage"] { padding: 0.2rem 0; }
        .stChatMessage { border-radius: 16px; margin-bottom: 0.5rem; }
        .stChatMessage [data-testid="stChatMessageContent"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 0.9rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AI Chatbot")
    provider = os.getenv("CHAT_PROVIDER", "openai").strip().lower()
    provider_label = "Ollama local model" if provider == "ollama" else "OpenAI"
    st.caption(f"Powered by {provider_label}. Replies stream as they are generated.")
    if "messages" not in st.session_state:
        st.session_state.messages = [("assistant", WELCOME_MESSAGE)]

    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    if prompt := st.chat_input("Ask me anything"):
        previous_messages = list(st.session_state.messages)
        st.session_state.messages.append(("user", prompt))
        with st.chat_message("assistant"):
            try:
                response = st.write_stream(stream_chat_reply(prompt, previous_messages))
            except ChatbotError as exc:
                st.error(str(exc))
                response = None
        if response:
            st.session_state.messages.append(("assistant", response))

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = [("assistant", WELCOME_MESSAGE)]
        st.rerun()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
