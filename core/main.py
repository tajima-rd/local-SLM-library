import os
import construct # type: ignore
import chain
import rag
from pathlib import Path
from langchain_ollama import OllamaLLM # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore


# --- ディレクトリ設定 ---
# 現在のファイル（スクリプト）のパス
current_path = Path(__file__).resolve()
 
# 'core' ディレクトリを含む親ディレクトリを見つける
core_root = next(p for p in current_path.parents if p.name == "core")

# そこから目的のサブパスを定義
base_dir = core_root / "sample"
markdown_dir = base_dir / "markdown"
vectorstore_dir = base_dir / "vectorstore"

# --- 入力ファイル ---
input_files = [
    markdown_dir / "cat_tools.md",
    markdown_dir / "japan_catapillar.md",
    markdown_dir / "Proffesional_College_of_arts_and_tourism.md",
]

# --- カテゴリ指定 ---
category = "test"

# --- ベクトルストアの構築 ---
for in_file in input_files:
    try:
        md_path = rag.prepare_markdown(in_file, markdown_dir / in_file.name)  # 再利用 or 変換先
    except ValueError as e:
        print(f"⚠️ スキップ: {e}")
        continue
    basename = md_path.stem
    vect_path = vectorstore_dir / f"{basename}.faiss"
    construct.vectorization(md_path, vect_path, category=category)


# --- モデル定義 ---
llm = OllamaLLM(model="granite3.3:8b")

CUSTOM_PROMPT = ChatPromptTemplate.from_template("""
        以下の情報に基づいて、**日本語で**専門的かつ論理的に、簡潔に答えてください。

        文脈:
        {context}

        質問:
        {input}
    """)

# --- チェーンの構築（カテゴリ指定） ---
qa_chain = chain.prepare_chain_for_category(
    llm=llm,
    category=category,
    base_path=vectorstore_dir,
    chain_type="conversational",
    prompt_template=CUSTOM_PROMPT,
    k=5,
)

# --- 対話ループ ---
print("RAG 実験モード開始（終了するには 'exit' と入力）")
while True:
    query = input("\n🗨 質問してください: ")
    if query.strip().lower() == "exit":
        break

    response = qa_chain.invoke({
        "input": query,
        "chat_history": []
    })

    for doc in response.get("source_documents", []):
        print("📄", doc.metadata.get("source"))

    print("\n🧠 回答:")
    print(response.get("answer") or response.get("output"))
