# Structured LLM Prompt Pipeline, JSON Extractor & Document Q&A

A Python CLI for extracting validated JSON from invoices or resumes, answering questions from indexed PDF or text documents, running local autonomous tool workflows, and serving a lightweight Streamlit chatbot.

## What it demonstrates

- Task 1: parameterized, task-specific prompt templates with a few-shot example
- Task 1: Pydantic schemas with required fields and field-level validation
- Task 1: an OpenAI structured-output provider and an offline deterministic provider
- Task 2: a Retrieval-Augmented Generation (RAG) pipeline using ChromaDB and OpenAI embeddings
- Task 2: PDF/text loading, recursive chunking, persistent local indexing, and source-page citations
- Task 3: a lightweight autonomous agent that can calculate, read files, and inspect the workspace
- Task 4: an interactive Streamlit chatbot with conversation memory
- Helpful validation errors and machine-readable JSON output
- Sample documents and automated tests

## Quick start

```powershell
cd structured-llm-json-extractor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m extractor samples/invoice.txt --document-type invoice
```

The command defaults to **offline mode**, so the example runs without credentials.

## Using OpenAI

```powershell
Copy-Item .env.example .env
# Edit .env and add OPENAI_API_KEY
python -m extractor samples/invoice.txt --document-type invoice --provider openai
```

The OpenAI provider requests a JSON schema response and validates it again locally with Pydantic. Set `OPENAI_MODEL` in `.env` to choose a compatible model.

## Task 2: Document Q&A with RAG

RAG defaults to OpenAI embeddings and answer generation; ChromaDB stores the vector index locally in `.rag-index/`. OpenAI API usage requires an account with available API credits. If the API reports exhausted credits, add billing/credits or use `--provider offline`. ChromaDB requires SQLite 3.35 or newer; if your Python runtime bundles an older SQLite version, install a newer Python and recreate the virtual environment. For OpenAI mode, add a valid `OPENAI_API_KEY` to `.env`; optionally set `OPENAI_MODEL` and `OPENAI_EMBEDDING_MODEL` (defaults: `gpt-4o-mini` and `text-embedding-3-small`).

On Windows, create a Python 3.11 environment for RAG if your current environment is older:

```powershell
py -3.11 -m venv .venv-rag
.\.venv-rag\Scripts\python.exe -m pip install -e ".[dev,rag]"
```

Index one or more PDF or UTF-8 text documents and ask a question in the same command:

```powershell
.\.venv-rag\Scripts\python.exe -m extractor rag samples/invoice.txt --question "What is the total due?"
```

For a no-API, no-credit local demo, add `--provider offline`:

```powershell
python -m extractor rag samples\invoice.txt --question "What is the total due?" --provider offline
```

Offline mode uses deterministic local lexical embeddings for retrieval and returns the best-matching source line with a citation. This is suitable for trying the pipeline without API access, but is less capable than OpenAI embeddings and generated answers. Offline and OpenAI indexes are stored in separate Chroma collections.

Index documents first, then ask questions in later runs:

```powershell
.\.venv-rag\Scripts\python.exe -m extractor rag samples/invoice.txt samples/resume.txt
.\.venv-rag\Scripts\python.exe -m extractor rag --question "Which skills are listed on the resume?"
```

Use `--persist-directory PATH` to select a different local index, and `--top-k N` to control how many chunks are retrieved. OpenAI answers are instructed to rely only on retrieved excerpts and cite them; the CLI also prints the source file and page for each retrieved excerpt. PDF page citations are one-based; text files are treated as page 1. OpenAI mode sends document content to OpenAI for embeddings and question answering; offline mode processes it locally. The local vector index is ignored by Git.

## Task 3: Multi-Tool Autonomous Agent

The project includes a lightweight local agent that can interpret arithmetic and file-based requests without external APIs.

```powershell
pip install -e ".[dev]"
python -m extractor agent "Add 12 and 8, then multiply by 2"
python -m extractor agent "Read file samples/invoice.txt"
```

This demonstrates a simple tool-calling loop: calculation, file listing, and local file reading.

## Task 4: Full-Stack AI Chatbot with Streamlit

Install the chat extra and launch the app:

```powershell
pip install -e ".[dev,chat]"
# Configure OPENAI_API_KEY and OPENAI_MODEL in the project .env file.
python -m extractor chatbot
# or, from the repo root:
streamlit run src/extractor/chatbot.py
```

The chatbot sends conversation history to the configured model and streams generated tokens into the page. With `CHAT_PROVIDER=openai`, a valid API key and available API credits are required.

To chat without API credits, install [Ollama](https://ollama.com/download), download a local model, and select it in `.env`:

```powershell
ollama pull llama3.2
# Set CHAT_PROVIDER=ollama in .env, then restart the app.
python -m extractor chatbot
```

Ollama must be running locally (by default at `http://localhost:11434`). Change `OLLAMA_MODEL` to use another model you have downloaded. Provider, connection, and configuration failures are shown in the chat instead of being replaced with canned fallback responses.

## Usage

```text
python -m extractor INPUT_FILE --document-type {invoice,resume} [--provider offline|openai] [--output result.json]
python -m extractor rag [PDF_OR_TXT ...] [--question QUESTION] [--persist-directory PATH] [--top-k N] [--provider openai|offline]
python -m extractor agent "YOUR TASK"
python -m extractor chatbot
```

The original extraction command defaults to offline mode and does not send content anywhere. RAG defaults to OpenAI; use `--provider offline` to keep document processing local.
