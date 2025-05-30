はい、承知いたしました。更新された `README.md` の内容を以下に表示します。このテキストをコピーして、お使いのコンピュータに `README.md` という名前でファイルとして保存してください。

```markdown
# 🧠 Hierarchical RAG System with LangChain × Ollama (and other tools)

This repository implements a modular, category-aware Retrieval-Augmented Generation (RAG) system built with [LangChain](https://www.langchain.com/) and [Ollama](https://ollama.com/). It supports hierarchical metadata tagging, vector store merging, and dynamic prompt/configuration switching for Japanese and English use cases. The repository also includes a separate module for decision tree analysis and simulation.

## 🚀 Features

- 📁 Category-based vector store construction using Markdown and other files
- 🧠 Conversational RAG chains with chat history context
- 🏷️ Metadata-aware document retrieval via FAISS and potentially database
- 🧩 Modular chain factory for various RAG strategies (`stuff`, `map_reduce`, etc.)
- 🧪 CLI for experimentation (`rag` or `llm` mode)
- 📎 Document support: Markdown (fully), DOCX/PDF via Docling integration
- 📐 Dynamic chunking via document structure analysis
- ✨ Database integration (e.g., for metadata, history)
- 🔄 Dynamic RAG configuration switching

*Note: Features specifically related to the `decisiontree` module are not listed here.*

## 📂 Project Structure

```
.
README.md                    # This file.
LICENSE.txt                  # MIT License.
requirements.txt             # External libraries.
core/                        # Core objects for RAG construction.
 ├── chain_factory.py        # Chain builder for LangChain RAG
 ├── database/               # Directory for database files
 ├── database.py             # Database utilities and interface
 ├── document_utils.py       # Document loaders and converters (Markdown, PDF, DOCX)
 ├── embedding_config.py     # Embedding model selection (via Ollama API)
 ├── llm_config.py           # LLM and prompt configuration
 ├── main.py                 # Main entry point to run the RAG CLI
 ├── process_utils.py        # Document processing (conversion, vectorization)
 ├── prompts.py              # Prompt templates (Japanese/English)
 ├── rag_session.py          # RAG session manager class (formerly RAGSession.py)
 ├── retriever_utils.py      # FAISS retriever manager and metadata editing
 ├── sample/                 # Sample data for RAG.
 │   ├── database.db         # Sample SQLite database file
 │   ├── markdown/           # Input Markdown files
 │   ├── pdf/                # Input PDF files (used with Docling)
 │   └── vectorstore/        # Output FAISS vectorstores (structured by category and document/section)
 └── switch_rag_objects.py   # Logic for dynamically switching RAG configurations or sources
decisiontree/                # Separate module for Decision Tree analysis and simulation.
 ├── model.py                # Defines decision tree structure and logic
 ├── parser.py               # Parses decision tree models from input files
 ├── sample/                 # Sample data for the decisiontree module
 │   ├── model/              # Sample decision tree model files
 │   ├── scenario/           # Sample scenario data for simulations
 │   └── simulated_results/  # Output directory for simulation results
 ├── sample.py               # Example script demonstrating decisiontree usage
 ├── simulator.py            # Handles the simulation execution
 └── utils.py                # Utility functions for the decisiontree module
```
*Note: The `__pycache__` directories and standard `__init__.py` files are omitted for brevity.*

## 🔧 Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- LangChain and community extensions
- Docling (for document conversion like PDF/DOCX)
- Dependencies for the `decisiontree` module (check `requirements.txt`)

Install dependencies:

```bash
pip install -r requirements.txt
````

Install models:

```bash
ollama pull nomic-embed-text:latest
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
python core/main.py
```

When prompted:

* Type a question (in Japanese or English depending on prompt)
* Type `exit` to quit the session

*Note: Usage instructions for the `decisiontree` module are not covered in this RAG-focused README.*

This metadata is used for selecting relevant documents by `tagname` and `level`, facilitating the hierarchical retrieval.

## 📚 Prompt Options

You can choose from various prompt styles defined in `core/llm_config.py`:

*   `japanese_concise` – 論理的かつ簡潔な日本語応答
*   `default` – 簡易日本語
*   `english_verbose` – Detailed English response
*   `rephrase` – Search query rephrasing

## 📌 Notes

*   `granite-embedding:278m` is known to crash during vectorization. Use `bge-m3` or `nomic-embed-text:latest`.
*   Only `.md` is natively supported for direct loading. PDF/DOCX will be converted via Docling integration.
*   The `decisiontree/` directory contains a separate module unrelated to the core RAG system described in the main body of this README. Refer to its specific documentation or sample scripts (`decisiontree/sample.py`) for usage details.

## 📜 License

MIT License (c) 2025 Yu Fujimoto
```
