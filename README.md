# 🧠 Hierarchical RAG System with LangChain × Ollama

This repository implements a modular, category-aware Retrieval-Augmented Generation (RAG) system built with [LangChain](https://www.langchain.com/) and [Ollama](https://ollama.com/). It supports hierarchical metadata tagging, vector store merging, and dynamic prompt switching for Japanese and English use cases.

## 🚀 Features

- 📁 Category-based vector store construction using Markdown files
- 🧠 Conversational RAG chains with chat history context
- 🏷️ Metadata-aware document retrieval via FAISS
- 🧩 Modular chain factory for various RAG strategies (`stuff`, `map_reduce`, etc.)
- 🧪 CLI for experimentation (`rag` or `llm` mode)
- 📎 Document support: Markdown (fully), DOCX/PDF via Docling
- 📐 Dynamic chunking via document structure analysis

## 📂 Project Structure

```

.
├── main.py                  # Main entry point to run the RAG CLI
├── RAGSession.py           # Session manager class
├── chain\_factory.py        # Chain builder for LangChain RAG
├── retriever\_utils.py      # FAISS retriever manager and metadata editing
├── process\_utils.py        # Wrapper for document conversion + vectorization
├── vectorization.py        # Vector store generation logic
├── document\_utils.py       # Document loaders and Markdown converters
├── llm\_config.py           # LLM and prompt configuration
├── embedding\_config.py     # Embedding model selection (via Ollama API)
├── interactive\_cli.py      # CLI mode runner (RAG and LLM)
├── prompts.py              # Prompt templates (Japanese/English)
└── sample/
├── markdown/           # Input Markdown files
└── vectorstore/        # Output FAISS vectorstores

````

## 🔧 Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- LangChain and community extensions
- Docling (for document conversion)

Install dependencies:

```bash
pip install -r requirements.txt
````

Install models:

```bash
ollama pull bge-m3
ollama pull gemma3:4b
````

Start Ollama backend (in another terminal):

```bash
ollama serve
```

## ▶️ Usage

Run in interactive RAG mode:

```bash
python main.py
```

When prompted:

* Type a question (in Japanese or English depending on prompt)
* Type `exit` to quit the session

## 🏷️ Metadata Structure

Each `.faiss` vector store is accompanied by a `metadata.json`:

```json
{
  "embedding_model": "nomic-embed-text:latest",
  "loader_type": "markdown",
  "text_splitter_type": "recursive",
  "category": {
    "tagname": "大学",
    "level": 1
  }
}
```

This metadata is used for selecting relevant documents by `tagname` and `level`.

## 📚 Prompt Options

You can choose from various prompt styles in `llm_config.py`:

* `japanese_concise` – 論理的かつ簡潔な日本語応答
* `default` – 簡易日本語
* `english_verbose` – Detailed English response
* `rephrase` – Search query rephrasing

## 📌 Notes

* `granite-embedding:278m` is known to crash during vectorization. Use `nomic-embed-text:latest`.
* Only `.md` is natively supported. PDF/DOCX will be converted via Docling.

## 📜 License

MIT License (c) 2025 Yu Fujimoto
