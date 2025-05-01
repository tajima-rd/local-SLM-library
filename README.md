# local-SLM-library

ローカル環境で RAG（Retrieval-Augmented Generation）を実装・検証するための Python プロジェクトです。LangChain + Ollama を基盤とし、カテゴリベースのベクトルストア管理と柔軟なプロンプト設計が特徴です。

## 📦 主な機能

- Markdown / PDF / Word / PPTX ファイルをベクトル化し、カテゴリ別に管理
- 複数のベクトルストアをマージして統合検索
- LangChain 0.3 系に対応した Conversational Retrieval QA 構築
- 日本語・多言語対応（Granite 3.2:8b で確認済み）
- PromptTemplate による柔軟な対話設計
- docling による高度な文書変換パイプライン（Markdown 変換）

## 🧰 使用技術

- Python 3.11+
- [LangChain](https://python.langchain.com/)
- [Ollama](https://ollama.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [docling](https://github.com/docling-ai/docling)
- Unstructured / PyMuPDF など

## 📁 ディレクトリ構成

```
modules/core/
├── main.py            # 実行スクリプト
├── chain.py           # チェーン構築とプロンプト設計
├── rag.py             # ベクトルストア処理と統合
├── construct.py       # 埋め込み生成とMarkdown変換
├── prompts.py         # プロンプトテンプレート定義
└── sample/            # サンプルデータ（Markdown・ベクトルストア）
```

## 🚀 セットアップ手順

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

または、必要に応じて以下を手動でインストール：

```bash
pip install langchain langchain-community langchain-ollama pydantic faiss-cpu unstructured
```

### 2. Ollama モデルの取得（例）

```bash
ollama pull granite3.2:8b
ollama serve
```

### 3. 実行

```bash
python modules/core/main.py
```

## 📚 使用方法の概要

### 1. ベクトルストアの構築（construct.vectorization）

```python
vectorization(
    in_file="path/to/file.pdf",
    md_path="path/to/output.md",
    vect_path="path/to/vectorstore",
    category="example"
)
```

### 2. QAチェーンの準備（カテゴリベース）

```python
qa_chain = prepare_chain_for_category(
    llm=my_llm,
    category="example",
    base_path=Path("path/to/vectorhouse"),
    chain_type="conversational",
    prompt_template=CUSTOM_PROMPT
)
```

### 3. 対話実行

```python
response = qa_chain.invoke({
    "input": "質問内容",
    "chat_history": []
})
```

## 🧠 カスタムプロンプト例

```python
from langchain.prompts import ChatPromptTemplate

CUSTOM_PROMPT = ChatPromptTemplate.from_template("""
以下の文脈に基づいて、正確かつ論理的な日本語で回答してください。

文脈:
{context}

質問:
{input}
""")
```

## ⚠️ 注意点

- Granite Embedding 278m モデルは一部環境でクラッシュの報告があります（`nomic-embed-text:latest` を推奨）
- LangChain のバージョンは `0.3.x` に固定してあります（以降の互換性は未検証）
- Pydantic v2 系に対応

## 📝 ライセンス

MIT License

## 📬 開発者

Yu Fujimoto｜Tajima R&D
GitHub: [@tajima-rd](https://github.com/tajima-rd)
```

---

ご希望に応じて、英語版や図付きの README も用意可能です。必要ですか？
