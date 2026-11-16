import json
from types import SimpleNamespace

import httpx
import pytest

from extractor import chatbot


def test_stream_chat_reply_sends_conversation_and_yields_response(monkeypatch):
    observed = {}
    history = [
        turn
        for index in range(12)
        for turn in (("user", f"Question {index}"), ("assistant", f"Answer {index}"))
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            observed.update(kwargs)
            return iter(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]
                    ),
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=" there!"))]
                    ),
                ]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHAT_PROVIDER", "openai")
    monkeypatch.setattr(chatbot, "_create_client", lambda api_key: fake_client)

    answer = chatbot.build_chatbot_reply(
        "How are you?",
        history,
    )

    assert answer == "Hello there!"
    assert observed["stream"] is True
    assert len(observed["messages"]) == 26
    assert observed["messages"][1] == {"role": "user", "content": "Question 0"}
    assert observed["messages"][-2] == {"role": "assistant", "content": "Answer 11"}
    assert observed["messages"][-1] == {"role": "user", "content": "How are you?"}
    assert observed["max_tokens"] == 800


def test_chat_reply_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CHAT_PROVIDER", "openai")

    with pytest.raises(chatbot.ChatbotError, match="No OpenAI API key is configured"):
        chatbot.build_chatbot_reply("Hello")


def test_ollama_chat_does_not_require_an_openai_key(monkeypatch):
    observed = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield json.dumps({"message": {"content": "Local answer"}})
            yield json.dumps({"message": {"content": " from Ollama"}})

    def fake_stream(method, url, **kwargs):
        observed.update({"method": method, "url": url, **kwargs})
        return FakeResponse()

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CHAT_PROVIDER", "ollama")
    monkeypatch.setattr(httpx, "stream", fake_stream)

    answer = chatbot.build_chatbot_reply("What is a local model?")

    assert answer == "Local answer from Ollama"
    assert observed["method"] == "POST"
    assert observed["url"] == "http://localhost:11434/api/chat"
    assert observed["json"]["model"] == "llama3.2:1b"
    assert observed["json"]["stream"] is True
    assert observed["json"]["keep_alive"] == "15m"
    assert observed["json"]["options"]["num_ctx"] == 8192
    assert observed["json"]["options"]["num_predict"] == 256
