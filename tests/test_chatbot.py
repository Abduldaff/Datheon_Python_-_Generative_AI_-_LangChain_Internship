from types import SimpleNamespace

import pytest

from extractor import chatbot


def test_stream_chat_reply_sends_conversation_and_yields_response(monkeypatch):
    observed = {}

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
    monkeypatch.setattr(chatbot, "_create_client", lambda provider, api_key: fake_client)

    answer = chatbot.build_chatbot_reply(
        "How are you?",
        [("user", "Hi"), ("assistant", "Hello!")],
    )

    assert answer == "Hello there!"
    assert observed["stream"] is True
    assert observed["messages"][-1] == {"role": "user", "content": "How are you?"}
    assert observed["messages"][-2] == {"role": "assistant", "content": "Hello!"}


def test_chat_reply_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CHAT_PROVIDER", "openai")

    with pytest.raises(chatbot.ChatbotError, match="No OpenAI API key is configured"):
        chatbot.build_chatbot_reply("Hello")


def test_ollama_chat_does_not_require_an_openai_key(monkeypatch):
    observed = {}

    class FakeCompletions:
        def create(self, **kwargs):
            observed.update(kwargs)
            return iter(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Local answer"))]
                    )
                ]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CHAT_PROVIDER", "ollama")
    monkeypatch.setattr(chatbot, "_create_client", lambda provider, api_key: fake_client)

    answer = chatbot.build_chatbot_reply("What is a local model?")

    assert answer == "Local answer"
    assert observed["model"] == "llama3.2"
    assert observed["stream"] is True
