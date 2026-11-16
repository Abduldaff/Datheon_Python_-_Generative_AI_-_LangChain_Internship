from __future__ import annotations

import json
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
    "directly, naturally, and accurately. Use the complete prior conversation for context. "
    "Lead with the answer and be concise unless the user asks for detail. If unsure, say so "
    "rather than inventing facts."
)
WELCOME_MESSAGE = (
    "Hi! I’m your AI assistant. Ask me anything to start a conversation."
)


class ChatbotError(RuntimeError):
    """A user-facing error raised when the chatbot cannot get a model response."""


def _create_client(api_key: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ChatbotError(
            "The chatbot client SDK is not installed. Run: pip install -e \".[chat]\""
        ) from exc
    return OpenAI(api_key=api_key)


def _stream_ollama(model: str, messages: list[dict[str, str]]) -> Iterator[str]:
    try:
        import httpx
    except ImportError as exc:
        raise ChatbotError(
            "The HTTP client is not installed. Run: pip install -e \".[chat]\""
        ) from exc

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    request = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": "15m",
        "options": {
            "temperature": 0.4,
            "num_ctx": 8192,
            "num_predict": 256,
        },
    }

    try:
        with httpx.stream(
            "POST",
            f"{base_url}/api/chat",
            json=request,
            timeout=httpx.Timeout(120.0, connect=5.0),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ChatbotError("Ollama returned an invalid streaming response.") from exc
                content = event.get("message", {}).get("content")
                if content:
                    yield content
    except ChatbotError:
        raise
    except httpx.ConnectError as exc:
        raise ChatbotError(
            "Cannot connect to Ollama. Start Ollama and make sure its local server is running."
        ) from exc
    except httpx.TimeoutException as exc:
        raise ChatbotError("Ollama took too long to respond. Try again.") from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise ChatbotError(
                f"Ollama model '{model}' was not found. Download it with: ollama pull {model}"
            ) from exc
        raise ChatbotError(
            f"Ollama returned HTTP {exc.response.status_code}. Check the local Ollama server."
        ) from exc
    except httpx.HTTPError as exc:
        raise ChatbotError("The connection to Ollama failed while streaming the response.") from exc


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
            model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in history or []:
        if role in {"user", "assistant"} and content and content.strip():
            messages.append({"role": role, "content": content.strip()})
    if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != cleaned:
        messages.append({"role": "user", "content": cleaned})

    model = (
        os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        if provider == "ollama"
        else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    if provider == "ollama":
        yield from _stream_ollama(model, messages)
        return

    client = _create_client(api_key)
    try:
        stream = client.chat.completions.create(
            model=model,
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
    st.caption(
        f"Powered by {provider_label}. Full chat history is retained; replies stream as generated."
    )
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
